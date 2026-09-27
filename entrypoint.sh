#!/usr/bin/env bash
set -euo pipefail

# Ensure models folders exist (needed if mounted as empty volumes)
mkdir -p /app/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m
mkdir -p /app/ComfyUI/models/Trellis2

# Download DINOv3 model if missing
DINO_DIR="/app/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m"
if [ ! -f "${DINO_DIR}/model.safetensors" ]; then
    echo "DINOv3 model missing. Downloading..."
    curl -L -o "${DINO_DIR}/model.safetensors" "https://huggingface.co/PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m/resolve/main/model.safetensors"
fi
if [ ! -f "${DINO_DIR}/config.json" ]; then
    curl -L -o "${DINO_DIR}/config.json" "https://huggingface.co/PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m/resolve/main/config.json"
fi
if [ ! -f "${DINO_DIR}/preprocessor_config.json" ]; then
    curl -L -o "${DINO_DIR}/preprocessor_config.json" "https://huggingface.co/PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m/resolve/main/preprocessor_config.json"
fi

# Download Trellis2 GGUF models if missing
# We check if pipeline.json is present as an indicator of whether Trellis2 models are downloaded
if [ ! -f "/app/ComfyUI/models/Trellis2/pipeline.json" ]; then
    echo "Trellis2 models missing. Starting model downloader..."
    export COMFY_ROOT="/app"
    bash /app/trellis2-gguf-model-downloader.sh
fi

# Install ComfyUI-Manager if missing
if [ ! -d "/app/ComfyUI/custom_nodes/ComfyUI-Manager" ]; then
    echo "ComfyUI-Manager not found. Installing..."
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git /app/ComfyUI/custom_nodes/ComfyUI-Manager
    # Manager typically installs its own requirements on boot, but we can ensure it here
    python3 -m pip install -r /app/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt || true
fi

# Apply dynamic memory-efficient sliced attention patches on startup
echo "Applying custom sparse attention and naive backend patches..."
python3 - << 'EOF'
import os

def patch_file(path, replacements):
    if not os.path.exists(path):
        return
    content = open(path).read()
    for old, new in replacements:
        if new in content:
            continue
        if old in content:
            content = content.replace(old, new)
    open(path, 'w').write(content)

# 0. __init__.py
p_init = "/app/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/__init__.py"
replacements_init = [
    (
        '    print(f"[Trellis2-GGUF] Warning: Failed to monkeypatch DinoV3ProjFeatureExtractor.forward: {e}")',
        """    if not (isinstance(e, ImportError) and "trellis2" in str(e)):
        print(f"[Trellis2-GGUF] Warning: Failed to monkeypatch DinoV3ProjFeatureExtractor.forward: {e}")"""
    ),
    (
        '    print(f"[Trellis2-GGUF] Warning: Failed to monkeypatch Trellis2ImageTo3DPipeline.get_proj_cond_shape: {e}")',
        """    if not (isinstance(e, ImportError) and "trellis2" in str(e)):
        print(f"[Trellis2-GGUF] Warning: Failed to monkeypatch Trellis2ImageTo3DPipeline.get_proj_cond_shape: {e}")"""
    ),
    (
        '    if not (isinstance(e, ImportError) and "trellis2" in str(e)):\\n        print(f"[Trellis2-GGUF] Warning: Failed to monkeypatch DinoV3ProjFeatureExtractor.forward: {e}")',
        """    if not (isinstance(e, ImportError) and "trellis2" in str(e)):
        print(f"[Trellis2-GGUF] Warning: Failed to monkeypatch DinoV3ProjFeatureExtractor.forward: {e}")"""
    ),
    (
        '    if not (isinstance(e, ImportError) and "trellis2" in str(e)):\\n        print(f"[Trellis2-GGUF] Warning: Failed to monkeypatch Trellis2ImageTo3DPipeline.get_proj_cond_shape: {e}")',
        """    if not (isinstance(e, ImportError) and "trellis2" in str(e)):
        print(f"[Trellis2-GGUF] Warning: Failed to monkeypatch Trellis2ImageTo3DPipeline.get_proj_cond_shape: {e}")"""
    )
]
patch_file(p_init, replacements_init)



# 5. gguf_utils.py
p_gguf = '/app/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/trellis2_gguf/utils/gguf_utils.py'
replacements_gguf = [
    (
        '            torch_tensor = torch.from_numpy(tensor.data)',
        '            torch_tensor = torch.from_numpy(tensor.data).clone()'
    )
]
patch_file(p_gguf, replacements_gguf)

# 6. ComfyUI-GGUF/loader.py
p_loader = "/app/ComfyUI/custom_nodes/ComfyUI-GGUF/loader.py"
replacements_loader = [
    (
        "torch_tensor = torch.from_numpy(tensor.data) # mmap",
        "torch_tensor = torch.from_numpy(tensor.data).clone() # mmap (cloned to prevent segfaults on unload)"
    )
]
patch_file(p_loader, replacements_loader)

# 7. trellis2_image_to_3d.py unload synchronization and chunked rasterization
p_image3d = "/app/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/trellis2_gguf/pipelines/trellis2_image_to_3d.py"
if os.path.exists(p_image3d):
    import re
    content = open(p_image3d).read()
    if "torch.cuda.synchronize()" not in content:
        pattern = r"(def unload_[a-zA-Z0-9_]+\(self\):\s*if self\.models\['[a-zA-Z0-9_]+'\] is not None:)"
        replacement = r"\1\n            if torch.cuda.is_available(): torch.cuda.synchronize()"
        new_content = re.sub(pattern, replacement, content)
    else:
        new_content = content
    
    old_rast = "rast, _ = dr.rasterize(\n            ctx, uvs_torch, faces_torch,\n            resolution=[texture_size, texture_size],\n        )"
    new_rast = """faces_torch_int32 = faces_torch.to(torch.int32).contiguous()
        rast = torch.zeros((1, texture_size, texture_size, 4), device=vertices_torch.device, dtype=torch.float32)
        chunk_size = 50000
        for i in range(0, faces_torch_int32.shape[0], chunk_size):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_torch, faces_torch_int32[i:i+chunk_size],
                resolution=[texture_size, texture_size],
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i
            rast = torch.where(mask_chunk, rast_chunk, rast)"""
    if old_rast in new_content:
        new_content = new_content.replace(old_rast, new_rast)
        new_content = new_content.replace("pos = dr.interpolate(vertices_torch.unsqueeze(0), rast, faces_torch)[0][0]", "pos = dr.interpolate(vertices_torch.unsqueeze(0), rast, faces_torch_int32)[0][0]")
        
    if new_content != content:
        open(p_image3d, 'w').write(new_content)

# 8. comfy_extras/nodes_glsl.py: disable ANGLE EGL preloading on Linux to prevent Vulkan crashes
p_glsl = "/app/ComfyUI/comfy_extras/nodes_glsl.py"
if os.path.exists(p_glsl):
    content = open(p_glsl).read()
    old_code = "def _preload_angle():\n    egl_path = comfy_angle.get_egl_path()"
    new_code = "def _preload_angle():\n    if sys.platform == 'linux':\n        return\n    egl_path = comfy_angle.get_egl_path()"
    if old_code in content:
        open(p_glsl, 'w').write(content.replace(old_code, new_code))
EOF

echo "All required models checked."

# Ensure rembg and onnxruntime are installed and working
if ! python3 -c "import rembg; import onnxruntime" 2>/dev/null; then
    echo "Installing missing rembg and onnxruntime dependencies..."
    python3 -m pip install onnxruntime rembg
fi

# Ensure smart-uv-projection is installed for Smart UV unwrapping
if ! python3 -c "import smart_uv" 2>/dev/null; then
    echo "Installing smart-uv-projection for pure Python Smart UV unwrapping..."
    python3 -m pip install "https://github.com/Aero-Ex/Smart-UV-Projection/releases/download/v0.1.0/smart_uv_projection-0.1.0-py3-none-any.whl"
fi

echo "Starting ComfyUI..."

# Runtime variables for ROCm / gfx1200 performance
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Fix transformers/tokenizers version mismatch crashing ComfyUI on this PyTorch build
python3 -m pip install -U transformers huggingface_hub tokenizers meshlib
if [ -f "/app/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt" ]; then
    python3 -m pip install -r /app/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt
fi

# Ensure xvfb is installed for headless OpenGL rendering (fixes nvdiffrast segfaults)
if ! command -v Xvfb >/dev/null 2>&1; then
    echo "Installing xvfb for headless 3D rendering..."
    apt-get update && apt-get install -y xvfb
fi

# Clean stale X11 lock files and start Xvfb with auto-respawn in background
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
(
    while true; do
        rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
        Xvfb :99 -screen 0 1024x768x24 -nolisten tcp
        sleep 1
    done
) &
export DISPLAY=:99
sleep 1

# Launch ComfyUI
exec python3 -X faulthandler /app/ComfyUI/main.py --listen 0.0.0.0 --port 8188 --use-flash-attention --disable-pinned-memory "$@"
