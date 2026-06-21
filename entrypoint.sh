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

echo "All required models checked."
echo "Starting ComfyUI..."

# Runtime variables for ROCm / gfx1200 performance
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"

# Launch ComfyUI
exec python3 /app/ComfyUI/main.py --listen 0.0.0.0 --port 8188 --use-pytorch-cross-attention "$@"
