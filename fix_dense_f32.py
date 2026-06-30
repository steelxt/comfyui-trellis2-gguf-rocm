import os

dense_attn = "/app/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/trellis2_gguf/modules/attention/full_attn.py"

with open(dense_attn, "r") as f:
    fc = f.read()

# Patch _naive_sdpa in dense full_attn.py
fc = fc.replace(
    "    attn_weight = q @ k.transpose(-2, -1) * scale_factor",
    "    attn_weight = q.to(torch.float32) @ k.transpose(-2, -1).to(torch.float32) * scale_factor"
)

with open(dense_attn, "w") as f:
    f.write(fc)

print("Successfully patched dense float32 matmul!")
