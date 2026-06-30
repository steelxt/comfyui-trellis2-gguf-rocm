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

# 7. trellis2_image_to_3d.py unload synchronization
p_image3d = "/app/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF/trellis2_gguf/pipelines/trellis2_image_to_3d.py"
if os.path.exists(p_image3d):
    import re
    content = open(p_image3d).read()
    pattern = r"(def unload_[a-zA-Z0-9_]+\(self\):\s*if self\.models\['[a-zA-Z0-9_]+'\] is not None:)"
    replacement = r"\1\n            if torch.cuda.is_available(): torch.cuda.synchronize()"
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        open(p_image3d, 'w').write(new_content)
EOF


echo "All required models checked."

# Ensure rembg and onnxruntime are installed and working
if ! python3 -c "import rembg; import onnxruntime" 2>/dev/null; then
    echo "Installing missing rembg and onnxruntime dependencies..."
    python3 -m pip install onnxruntime rembg
fi

echo "Starting ComfyUI..."

# Runtime variables for ROCm / gfx1200 performance
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Launch ComfyUI
exec python3 -X faulthandler /app/ComfyUI/main.py --listen 0.0.0.0 --port 8188 --use-flash-attention --disable-pinned-memory "$@"
