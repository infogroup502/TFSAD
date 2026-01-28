import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def process(x,model='double',patch_size=10):
    B, L, M = x.shape
    n = L // patch_size
    x = rearrange(x, 'b l m -> b m l')
    x = rearrange(x, 'b m (p n) -> (b m) p n', p=patch_size, n=n)

    batch_size, rows, cols = x.shape
    device = x.device

    # --------------------------
    idx_local = torch.zeros((cols, cols), dtype=torch.long, device=device)
    for j in range(cols):

        other_cols = torch.arange(cols, device=device) != j
        idx_local[j] = torch.cat([torch.tensor([j], device=device), torch.arange(cols, device=device)[other_cols]])

    local_concat_results = x.unsqueeze(2)
    local_concat_results = local_concat_results.expand(-1, -1, cols, -1)
    local_concat_results = local_concat_results.gather(dim=3,
                                                       index=idx_local.unsqueeze(0).unsqueeze(0).expand(batch_size, rows,
                                                                                                        -1, -1))
    idx_global = torch.zeros((rows, rows), dtype=torch.long, device=device)
    for i in range(rows):

        other_rows = torch.arange(rows, device=device) != i
        idx_global[i] = torch.cat([torch.tensor([i], device=device), torch.arange(rows, device=device)[other_rows]])
    global_concat_temp = x.permute(0, 2, 1)
    global_concat_temp = global_concat_temp.unsqueeze(2)
    global_concat_temp = global_concat_temp.expand(-1, -1, rows, -1)
    global_concat_temp = global_concat_temp.gather(dim=3,
                                                   index=idx_global.unsqueeze(0).unsqueeze(0).expand(batch_size, cols, -1,
                                                                                                     -1))
    global_concat_results = global_concat_temp.permute(0, 2, 1, 3)

    if model == 'double':
        return local_concat_results,global_concat_results
    elif model == 'l':
        return local_concat_results

def process_neighbors(x, patch_size):
    B, L, M = x.shape
    local_size = patch_size
    global_size = patch_size
    local_neighbors = []
    global_neighbors = []

    for i in range(L):
        # ===== local: [i, i-(local_size-1), ..., i-1] =====
        # indices like: [i] + [i-local_size+1, ..., i-1]
        local_idx = [i] + list(range(i - local_size + 1, i))

        # clamp + pad: if idx < 0 -> 0
        local_idx = [0 if t < 0 else t for t in local_idx]
        local_part = x[:, local_idx, :]                 # (B, local_size, M)
        local_neighbors.append(local_part)

        # ===== global:=====
        # [i-local_size-global_size+1, ..., i-local_size]
        g_end = i - local_size
        g_start = g_end - global_size + 1
        global_idx = list(range(g_start, g_end + 1))

        # clamp + pad: if idx < 0 -> 0
        global_idx = [0 if t < 0 else t for t in global_idx]
        global_part = x[:, global_idx, :]               # (B, global_size, M)
        global_neighbors.append(global_part)

    x_local = torch.stack(local_neighbors, dim=1)       # (B, L, local_size, M)
    x_global = torch.stack(global_neighbors, dim=1)     # (B, L, global_size, M)

    return x_local.permute(0,1,3,2), x_global.permute(0,1,3,2)

def FFT(x, dim=-1, norm='ortho',return_magnitude=True):
    if x.is_complex():
        fft_result = torch.fft.fft(x, dim=dim, norm=norm)
    else:
        fft_result = torch.fft.rfft(x, dim=dim, norm=norm)

    if return_magnitude:
        return torch.abs(fft_result)
    else:
        return fft_result

def hosvd(X: torch.Tensor, ranks=(1, 1, 1)):

    B, L, M, N, P = X.shape
    r_m, r_n, r_p = ranks

    # mode-M unfolding: (M, B*L*N*P)
    Xm = X.permute(2, 0, 1, 3, 4).reshape(M, -1)
    # mode-N unfolding: (N, B*L*M*P)
    Xn = X.permute(3, 0, 1, 2, 4).reshape(N, -1)
    # mode-P unfolding: (P, B*L*M*N)
    Xp = X.permute(4, 0, 1, 2, 3).reshape(P, -1)

    # HOSVD: U^(n) = mode-n unfolding
    Um, _, _ = torch.linalg.svd(Xm, full_matrices=False)
    Un, _, _ = torch.linalg.svd(Xn, full_matrices=False)
    Up, _, _ = torch.linalg.svd(Xp, full_matrices=False)

    U_m = Um[:, :r_m].contiguous()
    U_n = Un[:, :r_n].contiguous()
    U_p = Up[:, :r_p].contiguous()
    return U_m, U_n, U_p

@torch.no_grad()
def decomposition(
    X: torch.Tensor,
    rank: int = 1,
    return_core: bool = False,
    check_orth: bool = False,
):
    B, L, M, N, P = X.shape
    U_m, U_n, U_p = hosvd(X, ranks=(rank, rank, rank))

    if check_orth:   # verify orthogonality
        I = torch.eye(rank, device=X.device, dtype=X.dtype)
        em = (U_m.T @ U_m - I).abs().max().item()
        en = (U_n.T @ U_n - I).abs().max().item()
        ep = (U_p.T @ U_p - I).abs().max().item()
        print(f"[orth err] m={em:.3e}, n={en:.3e}, p={ep:.3e}")

    # ---------------- (B,L,M)/(B,L,N)/(B,L,P) ----------------

    u_m = U_m[:, 0]
    u_n = U_n[:, 0]
    u_p = U_p[:, 0]

    # (B,L,M): ×_N u_n^T ×_P u_p^T
    tmp = torch.tensordot(X, u_n, dims=([3], [0]))      # (B,L,M,P)
    X_blm = torch.tensordot(tmp, u_p, dims=([3], [0]))  # (B,L,M)

    # (B,L,N): ×_M u_m^T ×_P u_p^T
    tmp = torch.tensordot(X, u_m, dims=([2], [0]))      # (B,L,N,P)
    X_bln = torch.tensordot(tmp, u_p, dims=([3], [0]))  # (B,L,N)

    # (B,L,P): ×_M u_m^T ×_N u_n^T
    tmp = torch.tensordot(X, u_m, dims=([2], [0]))      # (B,L,N,P)
    X_blp = torch.tensordot(tmp, u_n, dims=([2], [0]))  # (B,L,P)

    if not return_core:
        return X_blm.permute(0,2,1), X_bln.permute(0,2,1), X_blp.permute(0,2,1)

        # core: (B,L,1,1,1)
    G = X
    # G: (B,L,M,N,P) ×_M U_m^T => (B,L,1,N,P)
    G = torch.tensordot(G, U_m, dims=([2], [0]))        # (B,L,N,P,1)
    G = G.permute(0, 1, 4, 2, 3).contiguous()           # (B,L,1,N,P)
    # ×_N U_n^T => (B,L,1,1,P)
    G = torch.tensordot(G, U_n, dims=([3], [0]))        # (B,L,1,P,1)
    G = G.permute(0, 1, 2, 4, 3).contiguous()           # (B,L,1,1,P)
    # ×_P U_p^T => (B,L,1,1,1)
    G = torch.tensordot(G, U_p, dims=([4], [0]))        # (B,L,1,1,1)

    return X_blm, X_bln, X_blp, G

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
