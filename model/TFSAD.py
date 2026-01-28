import time
import torch
import torch.nn as nn
from einops import rearrange


from model.feature_conv import *
from model.RevIN import RevIN
from model.utils import *

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)
        return x

class TFSAD(nn.Module):
    def __init__(self, win_size, patch_size=10, batch_size=1, channel=6, device=None):
        super(TFSAD, self).__init__()
        self.patch_size = patch_size
        self.win_size = win_size
        self.channel = channel
        self.p_size = (self.patch_size//2)+1
        self.n = self.win_size/self.patch_size
        self.m_n_p = self.channel+self.patch_size+self.p_size

        self.batch_size = batch_size
        self.device = device

        self.num = [self.channel,self.channel, self.channel]
        self.revin_layer = RevIN(num_features=channel)
        # self.Self_Atten = SelfAttention(d_model=self.channel-self.channel//2+14)
        self.tcn_lm = TemporalConvNet(num_inputs=self.channel,num_channels=self.num,kernel_size=self.patch_size,dropout=0.3)
        self.tcn_ln = TemporalConvNet(num_inputs=self.patch_size,num_channels=[self.patch_size,2*self.patch_size,self.patch_size],kernel_size=self.patch_size,dropout=0.3)
        self.tcn_lp = TemporalConvNet(num_inputs=self.p_size,num_channels=[self.p_size,2*self.p_size,self.p_size],kernel_size=self.patch_size,dropout=0.3)
        self.tcn_l_d = TemporalConvNet(num_inputs=self.m_n_p,num_channels=[self.m_n_p,self.m_n_p,self.m_n_p],kernel_size=self.patch_size,dropout=0.3)

        self.Attn_lm = VariableAttentionConv(num_variables=self.channel,conv_out_channels=self.channel,kernel_size=3,reduction_ratio=4)
        self.Attn_lm_1 = VariableAttentionConv(num_variables=self.channel-self.channel//2+2,conv_out_channels=self.channel-self.channel//2+2,kernel_size=3,reduction_ratio=4)
        self.Attn_ln = VariableAttentionConv(num_variables=self.patch_size,conv_out_channels=self.patch_size,kernel_size=3,reduction_ratio=4)
        self.Attn_ln_1 = VariableAttentionConv(num_variables=self.patch_size-self.patch_size//2+2,conv_out_channels=self.patch_size-self.patch_size//2+2,kernel_size=3,reduction_ratio=4)
        self.Attn_lp = VariableAttentionConv(num_variables=self.p_size,conv_out_channels=self.p_size,kernel_size=3,reduction_ratio=4)
        self.Attn_lp_1 = VariableAttentionConv(num_variables=((self.patch_size//2+2)//2+2),conv_out_channels=((self.patch_size//2+2)//2+2),kernel_size=3,reduction_ratio=4)
        self.Attn_d = VariableAttentionConv(num_variables=self.m_n_p,conv_out_channels=self.m_n_p,kernel_size=3,reduction_ratio=4)
        self.Attn_d_1 = VariableAttentionConv(num_variables=self.win_size,conv_out_channels=self.win_size,kernel_size=3,reduction_ratio=2)

        self.point_mlp = None
        self.neighbor_mlp = None

        self.Conv_lm = Conv(kernel_size=(self.channel,1))
        self.Conv_lm_1 = Conv(kernel_size=(self.channel//2,1))
        self.Convln = Conv(kernel_size=(self.patch_size,1))
        self.Conv_1 = Conv(kernel_size=(self.patch_size//2,1))
        self.Conv_lp = Conv(kernel_size=(self.p_size,1))
        self.Conv_lp_1 = Conv(kernel_size=(self.p_size//2,1))
        self.Conv_d = Conv(kernel_size=(self.m_n_p,1))
        self.Conv_d_1 = Conv(kernel_size=(self.m_n_p//2+4,1))

    def forward(self, x):
        B, L, M = x.shape
        self.to(x.device)

        x_norm = self.revin_layer(x, 'norm')

        local_concat_results, global_concat_results = process(x_norm, patch_size=self.patch_size)
        local_concat_results, global_concat_results = process_neighbors(x_norm, patch_size=self.patch_size)

        b, patch, p, n = local_concat_results.shape
        local_neighbor = local_concat_results.reshape(b, patch * p, n)
        l_l_p = process(local_neighbor, model='l')
        local_neighbor = local_neighbor.reshape(self.batch_size,self.channel, self.win_size, n)
        back_neighbor = local_neighbor.reshape(self.batch_size, self.win_size, self.patch_size * self.channel)

        b, patch, p, n = global_concat_results.shape
        global_neighbor = global_concat_results.reshape(b, patch * p, n)  # (b*m,l,n)
        g_l_p = process(global_neighbor, model='l')

        local_f = FFT(l_l_p, dim=-1).reshape(self.batch_size, self.channel, self.win_size, self.patch_size,-1)
        global_f = FFT(g_l_p, dim=-1).reshape(self.batch_size, self.channel, self.win_size, self.patch_size,-1)

        local_lm, local_ln, local_lp = decomposition(local_f.permute(0, 2, 1, 3, 4))  # a:m,l; b:n,l; c:p,l
        global_lm, global_ln, global_lp = decomposition(global_f.permute(0, 2, 1, 3, 4))


        local_d = torch.cat((local_lm,local_ln,local_lp),dim=1) # b,m+n+p,l
        global_d = torch.cat((global_lm,global_ln,global_lp),dim=1)
        # 特征维度
        local_d_0 = self.Conv_d(local_d.unsqueeze(1)).squeeze(1) # b,1,l
        local_d_1 = self.Conv_d_1(local_d.unsqueeze(1)).squeeze(1) # b,m-m//2+13,l
        local_data = torch.cat((local_d_0,local_d_1),dim=1).permute(0,2,1) # b,m-m//2+14,l ->b,l, ,
        # print(local_data.shape)
        local_data = self.Attn_d_1(local_data)
        global_d_0 = self.Conv_d(global_d.unsqueeze(1)).squeeze(1)
        global_d_1 = self.Conv_d_1(global_d.unsqueeze(1)).squeeze(1)
        global_data = torch.cat((global_d_0,global_d_1),dim=1).permute(0,2,1)
        global_data = self.Attn_d_1(global_data)
        # 时间维度
        local_t_d = self.tcn_l_d(local_d)
        global_t_d = self.tcn_l_d(global_d)

        local_t_d = self.Attn_d(local_t_d).permute(0,2,1) # b,m+16,l -> b,l,m+16
        global_t_d = self.Attn_d(global_t_d).permute(0,2,1)

        local_d = torch.cat((local_data,local_t_d),dim=-1) # b,l,2m-m//2+30
        global_d = torch.cat((global_data,global_t_d),dim=-1)

        local_l_m = self.Conv_lm(local_lm.unsqueeze(1)).squeeze(1) 
        local_l_m_1 = self.Conv_lm_1(local_lm.unsqueeze(1)).squeeze(1) 
        local_l_m = torch.cat((local_l_m, local_l_m_1), dim=1)  
        local_l_m = self.Attn_lm_1(local_l_m) # b,m-m//2+2,l

        local_l_n = self.Conv_ln(local_ln.unsqueeze(1)).squeeze(1)
        local_l_n_1 = self.Conv_ln_1(local_ln.unsqueeze(1)).squeeze(1) # b,n,l -> b,6,l
        local_l_n = torch.cat((local_l_n, local_l_n_1), dim=1) # b,7,l
        local_l_n = self.Attn_ln_1(local_l_n)

        local_l_p = self.Conv_lp(local_lp.unsqueeze(1)).squeeze(1)
        local_l_p_1 = self.Conv_lp_1(local_lp.unsqueeze(1)).squeeze(1)
        local_l_p = torch.cat((local_l_p, local_l_p_1), dim=1) # b,5,l
        local_l_p = self.Attn_lp_1(local_l_p)

        # global
        global_l_m = self.Conv_lm(global_lm.unsqueeze(1)).squeeze(1)
        global_l_m_1 = self.Conv_lm_1(global_lm.unsqueeze(1)).squeeze(1)
        global_l_m = torch.cat((global_l_m, global_l_m_1), dim=1)
        global_l_m = self.Attn_lm_1(global_l_m)

        global_l_n = self.Conv_ln(global_ln.unsqueeze(1)).squeeze(1)
        global_l_n_1 = self.Conv_ln_1(global_ln.unsqueeze(1)).squeeze(1)
        global_l_n = torch.cat((global_l_n, global_l_n_1), dim=1)
        global_l_n = self.Attn_ln_1(global_l_n)

        global_l_p = self.Conv_lp(global_lp.unsqueeze(1)).squeeze(1)
        global_l_p_1 = self.Conv_lp_1(global_lp.unsqueeze(1)).squeeze(1)
        global_l_p = torch.cat((global_l_p, global_l_p_1), dim=1)
        global_l_p = self.Attn_lp_1(global_l_p)

        local_lm = self.Attn_lm(self.tcn_lm(local_lm)) # b,m,l
        local_ln = self.Attn_ln(self.tcn_ln(local_ln))
        local_lp = self.Attn_lp(self.tcn_lp(local_lp))
        global_lm = self.Attn_lm(self.tcn_lm(global_lm))
        global_ln = self.Attn_ln(self.tcn_ln(global_ln))
        global_lp = self.Attn_lp(self.tcn_lp(global_lp))

        local_f_m = torch.cat((local_lm,local_l_m),dim=1).permute(0,2,1) # b,2m-m//2+2,l
        global_f_m = torch.cat((global_lm,global_l_m),dim=1).permute(0,2,1)
        local_f_n = torch.cat((local_ln,local_l_n),dim=1).permute(0,2,1) # b,17,l
        global_f_n = torch.cat((global_ln,global_l_n),dim=1).permute(0,2,1)
        local_f_p = torch.cat((local_lp,local_l_p),dim=1).permute(0,2,1) # b,11,l
        global_f_p = torch.cat((global_lp,global_l_p),dim=1).permute(0,2,1)

        local_f = torch.cat([local_f_m,local_f_n,local_f_p],dim=-1) # b,l,2m-m//2+30
        global_f = torch.cat([global_f_m,global_f_n,global_f_p],dim=-1) # b,l,2m-m//2+30

        local_f_d = torch.cat((local_f,local_d),dim=-1).reshape(self.batch_size*self.win_size,-1)
        global_f_d = torch.cat((global_f,global_d),dim=-1).reshape(self.batch_size*self.win_size,-1)

        if self.point_mlp is None:
            local_input_size = local_f_d.shape[-1]
            global_input_size = global_f_d.shape[-1]

            self.point_mlp = MLP(local_input_size,128,self.channel).to(x.device)
            self.neighbor_mlp = MLP(global_input_size,128,self.patch_size*self.channel).to(x.device)

        local_point = self.point_mlp(local_f_d).reshape(self.batch_size,self.win_size,-1)
        global_point = self.point_mlp(global_f_d).reshape(self.batch_size,self.win_size,-1)
        local_re_neighbor = self.neighbor_mlp(local_f_d).reshape(self.batch_size,self.win_size,-1)
        global_re_neighbor = self.neighbor_mlp(global_f_d).reshape(self.batch_size,self.win_size,-1)

        return local_point, global_point,local_re_neighbor,global_re_neighbor,back_neighbor

