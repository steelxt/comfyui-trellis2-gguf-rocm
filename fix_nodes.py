import os

p_nodes = "/app/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/nodes.py"
with open(p_nodes, "r") as f:
    c = f.read()

bad_hack = """if backend in ('cuda', 'triton'): backend = 'sdpa'
        os.environ['ATTN_BACKEND'] = backend
        try:
            from .trellis2_gguf.modules.attention import config as attn_config
            attn_config.BACKEND = backend
        except:
            pass"""
            
c = c.replace(bad_hack, "os.environ['ATTN_BACKEND'] = backend")

with open(p_nodes, "w") as f:
    f.write(c)
