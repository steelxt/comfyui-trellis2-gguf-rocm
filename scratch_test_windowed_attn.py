import torch
import math
import time

def current_loop_implementation(qkv_feats, seq_lens, device):
    # Currently implemented in windowed_attn.py
    q, k, v = qkv_feats.unbind(dim=1)
    outs = []
    off = 0
    for n in range(len(seq_lens)):
        sl = seq_lens[n].item()
        q_i = q[off:off + sl].transpose(0, 1).unsqueeze(0)
        k_i = k[off:off + sl].transpose(0, 1).unsqueeze(0)
        v_i = v[off:off + sl].transpose(0, 1).unsqueeze(0)
        
        out_i = torch.nn.functional.scaled_dot_product_attention(
            q_i, k_i, v_i, dropout_p=0.0, is_causal=False
        )[0]
        outs.append(out_i.transpose(0, 1))
        off += sl
    return torch.cat(outs, dim=0)

def vectorized_implementation(qkv_feats, seq_lens, device):
    # Proposed vectorized implementation
    q, k, v = qkv_feats.unbind(dim=1) # each [M, H, C]
    num_windows = len(seq_lens)
    max_len = int(seq_lens.max().item())
    H, C = q.shape[1], q.shape[2]
    
    # Grid of indices to build mask and select
    valid = torch.arange(max_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1) # [num_windows, max_len]
    
    q_pad = torch.zeros(num_windows, max_len, H, C, dtype=q.dtype, device=device)
    k_pad = torch.zeros(num_windows, max_len, H, C, dtype=k.dtype, device=device)
    v_pad = torch.zeros(num_windows, max_len, H, C, dtype=v.dtype, device=device)
    
    q_pad[valid] = q
    k_pad[valid] = k
    v_pad[valid] = v
    
    q_pad = q_pad.transpose(1, 2) # [num_windows, H, max_len, C]
    k_pad = k_pad.transpose(1, 2)
    v_pad = v_pad.transpose(1, 2)
    
    # Broadcastable key padding mask
    mask = valid.unsqueeze(1).unsqueeze(2) # [num_windows, 1, 1, max_len]
    
    out_pad = torch.nn.functional.scaled_dot_product_attention(
        q_pad, k_pad, v_pad, attn_mask=mask, dropout_p=0.0, is_causal=False
    ) # [num_windows, H, max_len, C]
    
    out_pad = out_pad.transpose(1, 2) # [num_windows, max_len, H, C]
    return out_pad[valid]

# Setup mock data resembling a realistic windowed attention call
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Suppose we have 2000 windows of lengths ranging from 10 to 64
num_windows = 2000
seq_lens = torch.randint(10, 64, size=(num_windows,), device=device)
M = int(seq_lens.sum().item())
H = 12 # 12 heads
C = 64 # head dimension

print(f"Total tokens (M): {M}, Windows: {num_windows}, Max window length: {seq_lens.max().item()}")

qkv_feats = torch.randn(M, 3, H, C, dtype=torch.float16, device=device)

# Warmup
_ = current_loop_implementation(qkv_feats, seq_lens, device)
_ = vectorized_implementation(qkv_feats, seq_lens, device)

# Measure time for current loop
t0 = time.time()
out_loop = current_loop_implementation(qkv_feats, seq_lens, device)
t_loop = time.time() - t0
print(f"Loop implementation time: {t_loop * 1000:.2f} ms")

# Measure time for vectorized implementation
t0 = time.time()
out_vec = vectorized_implementation(qkv_feats, seq_lens, device)
t_vec = time.time() - t0
print(f"Vectorized implementation time: {t_vec * 1000:.2f} ms")
print(f"Speedup: {t_loop / t_vec:.1f}x")

# Check mathematical equivalence
all_close = torch.allclose(out_loop, out_vec, rtol=1e-3, atol=1e-3)
max_diff = (out_loop - out_vec).abs().max().item()
print(f"Are implementations mathematically close? {all_close} (Max diff: {max_diff})")
