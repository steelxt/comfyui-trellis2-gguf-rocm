import os

full_attn = "/app/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/trellis2_gguf/modules/sparse/attention/full_attn.py"
windowed_attn = "/app/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/trellis2_gguf/modules/sparse/attention/windowed_attn.py"

with open(full_attn, "r") as f:
    fc = f.read()

# Patch _sliced_sdpa in full_attn.py
fc = fc.replace(
    "            attn = torch.matmul(q_chunk, kt) * scale\n            attn = torch.softmax(attn.to(torch.float32), dim=-1).to(q.dtype)",
    "            attn = torch.matmul(q_chunk.to(torch.float32), kt.to(torch.float32)) * scale\n            attn = torch.softmax(attn, dim=-1).to(q.dtype)"
)

with open(full_attn, "w") as f:
    f.write(fc)

with open(windowed_attn, "r") as f:
    wc = f.read()

# Patch naive sdpa in windowed_attn.py
wc = wc.replace(
    "                attn = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale\n                attn = torch.softmax(attn.to(torch.float32), dim=-1).to(q_i.dtype)",
    "                attn = torch.matmul(q_i.to(torch.float32), k_i.transpose(-2, -1).to(torch.float32)) * scale\n                attn = torch.softmax(attn, dim=-1).to(q_i.dtype)"
)

with open(windowed_attn, "w") as f:
    f.write(wc)
    
print("Successfully patched float32 matmul!")
