import torch
import torch.nn as nn
import torch.nn.functional as F



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout
    ):
        super(TemporalBlock, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dropout = dropout
        self.ll_conv1 = nn.Conv1d(
            n_inputs,          # 输入通道数
            n_outputs,         # 输出通道数
            kernel_size,       # 卷积核大小
            stride=stride,     # 步长
            padding=padding,   # 填充大小
            dilation=dilation  # 膨胀系数
        )
        self.chomp1 = Chomp1d(padding)

        self.ll_conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.sigmoid = nn.Sigmoid()

    def net(self, x, block_num, params=None):
        layer_name = "ll_tc.ll_temporal_block" + str(block_num)
        if params is None:
            x = self.ll_conv1(x)
        else:
            x = F.conv1d(
                x,                                      # 输入张量
                weight=params[layer_name + ".ll_conv1.weight"],  # 卷积核权重
                bias=params[layer_name + ".ll_conv1.bias"],      # 偏置
                stride=self.stride,                     # 步长
                padding=self.padding,                   # 填充
                dilation=self.dilation                  # 膨胀系数
            )

        x = self.chomp1(x)
        x = F.leaky_relu(x)

        return x

    def init_weights(self):

        self.ll_conv1.weight.data.normal_(0, 0.01)
        self.ll_conv2.weight.data.normal_(0, 0.01)

    def forward(self, x, block_num, params=None):
        # 调用net方法得到输出
        out = self.net(x, block_num, params)
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_levels = len(num_channels)

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            setattr(
                self,
                "ll_temporal_block{}".format(i),
                TemporalBlock(
                    in_channels,       # 输入通道数
                    out_channels,      # 输出通道数
                    kernel_size,       # 卷积核大小
                    stride=1,          # 步长固定为1
                    dilation=dilation_size,  # 膨胀系数
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout    # dropout概率
                ),
            )


    def forward(self, x, params=None):
        for i in range(self.num_levels):
            temporal_block = getattr(self, "ll_temporal_block{}".format(i))
            x = temporal_block(x, i, params=params)
        return x

class TemporalConvNet_2D(nn.Module):
    def __init__(self, kernel_size=(2, 1), stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 0),
            dilation=(1, 1),
        )

    def forward(self, x):

        main = F.leaky_relu(self.conv(x))

        return F.leaky_relu(main)


class VariableAttention(nn.Module):

    def __init__(self, num_variables, reduction_ratio=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

        hidden_size = max(num_variables // reduction_ratio, 4)
        self.mlp = nn.Sequential(
            nn.Linear(num_variables, hidden_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_variables, bias=False)
        )


        self.register_buffer('scale_factor', torch.tensor(1.0))

    def forward(self, x):

        pooled = self.pool(x).squeeze(-1)
        attn_weights = torch.sigmoid(self.mlp(pooled)).unsqueeze(-1)

        return x.mul_(attn_weights) if x.is_leaf else x * attn_weights


class Conv1DModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.activation(x)

class VariableAttentionConv(nn.Module):

    def __init__(self, num_variables, conv_out_channels=None, kernel_size=3, reduction_ratio=4):
        super().__init__()
        conv_out_channels = conv_out_channels or num_variables

        self.conv = Conv1DModule(num_variables, conv_out_channels, kernel_size)
        self.var_attn = VariableAttention(conv_out_channels, reduction_ratio)

        self.channels_last = False

    def forward(self, x):
        if self.channels_last and x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        x = self.conv(x)
        return self.var_attn(x)

    def to_channels_last(self):
        self = self.to(memory_format=torch.channels_last)
        self.channels_last = True
        return self

