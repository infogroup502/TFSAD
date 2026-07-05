import torch
import torch.fft

def FFT(x, dim=-1, norm='ortho',return_magnitude=True):
    # 检查输入是否为实数
    if x.is_complex():
        # 如果是复数，使用常规FFT
        fft_result = torch.fft.fft(x, dim=dim, norm=norm)
    else:
        # 如果是实数，使用rfft减少计算量
        fft_result = torch.fft.rfft(x, dim=dim, norm=norm)

    # 如果需要返回实数幅度
    if return_magnitude:
        return torch.abs(fft_result)
    else:
        return fft_result


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


def progress(x,model):
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
    elif model == 'g':
        return global_concat_results


def hosvd_tucker_factors_last3(X: torch.Tensor, ranks=(1, 1, 1)):
    """
    对 X:(B,L,M,N,P) 仅在 (M,N,P) 三个模态上做 HOSVD/Tucker 因子估计，
    得到正交因子 U_m, U_n, U_p（列正交）。

    ranks: (r_m, r_n, r_p)
    返回:
      U_m: (M, r_m), U_n: (N, r_n), U_p: (P, r_p)
    """
    assert X.ndim == 5, "X must be 5D: (B,L,M,N,P)"
    B, L, M, N, P = X.shape
    r_m, r_n, r_p = ranks

    # mode-M unfolding: (M, B*L*N*P)
    Xm = X.permute(2, 0, 1, 3, 4).reshape(M, -1)
    # mode-N unfolding: (N, B*L*M*P)
    Xn = X.permute(3, 0, 1, 2, 4).reshape(N, -1)
    # mode-P unfolding: (P, B*L*M*N)
    Xp = X.permute(4, 0, 1, 2, 3).reshape(P, -1)

    # HOSVD: U^(n) = left singular vectors of mode-n unfolding
    Um, _, _ = torch.linalg.svd(Xm, full_matrices=False)
    Un, _, _ = torch.linalg.svd(Xn, full_matrices=False)
    Up, _, _ = torch.linalg.svd(Xp, full_matrices=False)

    U_m = Um[:, :r_m].contiguous()
    U_n = Un[:, :r_n].contiguous()
    U_p = Up[:, :r_p].contiguous()
    return U_m, U_n, U_p


@torch.no_grad()
def tucker_bl_projections(
    X: torch.Tensor,
    rank: int = 1,
    return_core: bool = False,
    check_orth: bool = False,
):
    """
    用 Tucker/HOSVD 得到的因子矩阵做投影（使用 U^T），保持 B,L 不分解：
      X: (B,L,M,N,P)

    rank=1:
      返回严格为 (B,L,M), (B,L,N), (B,L,P)

    rank>1:
      返回 (B,L,M,rank,rank), (B,L,N,rank,rank), (B,L,P,rank,rank)

    return_core=True 时，额外返回 core:
      G = X ×_M U_m^T ×_N U_n^T ×_P U_p^T -> (B,L,rank,rank,rank)
    """
    assert X.ndim == 5, "X must be 5D: (B,L,M,N,P)"
    B, L, M, N, P = X.shape

    # 1) 用 HOSVD/Tucker 得到因子矩阵（列正交）
    U_m, U_n, U_p = hosvd_tucker_factors_last3(X, ranks=(rank, rank, rank))

    if check_orth:
        I = torch.eye(rank, device=X.device, dtype=X.dtype)
        em = (U_m.T @ U_m - I).abs().max().item()
        en = (U_n.T @ U_n - I).abs().max().item()
        ep = (U_p.T @ U_p - I).abs().max().item()
        print(f"[orth err] m={em:.3e}, n={en:.3e}, p={ep:.3e}")

    # ---------------- rank == 1: 输出 (B,L,M)/(B,L,N)/(B,L,P) ----------------
    if rank == 1:
        u_m = U_m[:, 0]  # (M,)
        u_n = U_n[:, 0]  # (N,)
        u_p = U_p[:, 0]  # (P,)

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
        # 做法：依次把 M,N,P 投影成 rank 维（这里 rank=1）
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

    # ---------------- rank > 1: 输出带两个 rank 维的三组张量 ----------------

    # (B,L,M,rank,rank): ×_N U_n^T ×_P U_p^T
    tmp = torch.tensordot(X, U_n, dims=([3], [0]))          # (B,L,M,P,rank)
    tmp = tmp.permute(0, 1, 2, 4, 3).contiguous()           # (B,L,M,rank,P)
    X_blm = torch.tensordot(tmp, U_p, dims=([4], [0]))      # (B,L,M,rank,rank)

    # (B,L,N,rank,rank): ×_M U_m^T ×_P U_p^T
    tmp = torch.tensordot(X, U_m, dims=([2], [0]))          # (B,L,N,P,rank)
    tmp = tmp.permute(0, 1, 2, 4, 3).contiguous()           # (B,L,N,rank,P)
    X_bln = torch.tensordot(tmp, U_p, dims=([4], [0]))      # (B,L,N,rank,rank)

    # (B,L,P,rank,rank): ×_M U_m^T ×_N U_n^T
    tmp = torch.tensordot(X, U_m, dims=([2], [0]))          # (B,L,N,P,rank)
    tmp = tmp.permute(0, 1, 3, 4, 2).contiguous()           # (B,L,P,rank,N)
    X_blp = torch.tensordot(tmp, U_n, dims=([4], [0]))      # (B,L,P,rank,rank)

    if not return_core:
        return X_blm, X_bln, X_blp

    # core: G = X ×_M U_m^T ×_N U_n^T ×_P U_p^T -> (B,L,rank,rank,rank)
    G = X
    # ×_M U_m^T: (B,L,M,N,P) -> (B,L,rank,N,P)
    G = torch.tensordot(G, U_m, dims=([2], [0]))            # (B,L,N,P,rank)
    G = G.permute(0, 1, 4, 2, 3).contiguous()               # (B,L,rank,N,P)

    # ×_N U_n^T: (B,L,rank,N,P) -> (B,L,rank,rank,P)
    G = torch.tensordot(G, U_n, dims=([3], [0]))            # (B,L,rank,P,rank)
    G = G.permute(0, 1, 2, 4, 3).contiguous()               # (B,L,rank,rank,P)

    # ×_P U_p^T: (B,L,rank,rank,P) -> (B,L,rank,rank,rank)
    assert G.shape[4] == U_p.shape[0], (G.shape, U_p.shape)  # 防止维度再写错
    G = torch.tensordot(G, U_p, dims=([4], [0]))            # (B,L,rank,rank,rank)

    return X_blm, X_bln, X_blp, G
