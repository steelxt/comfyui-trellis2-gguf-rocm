# Use the official AMD ROCm PyTorch image as base
ARG BASE_IMAGE=rocm/pytorch:latest
FROM ${BASE_IMAGE}

# Set non-interactive debian frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# nvdiffrast GL plugin compilation requires EGL/GL headers and libraries.
# iproute2 and lsof are used by the install script checks.
# OpenCV/Open3D runtime requires rendering libraries (libxrender1, libxi6, libxkbcommon0, libsm6, libglib2.0-0).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    cmake \
    lsof \
    iproute2 \
    ca-certificates \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgbm-dev \
    libxrender1 \
    libxi6 \
    libxkbcommon0 \
    libsm6 \
    libglib2.0-0 \
    libsparsehash-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

# Install ComfyUI python dependencies
RUN pip install --no-cache-dir -r /app/ComfyUI/requirements.txt

# Copy our repository files (helper scripts, json files, entrypoint) to /app
COPY . /app/

# Make scripts executable
RUN chmod +x /app/install-trellis2-gguf-rocm.sh /app/entrypoint.sh

# GPU Architecture build target & Compiler Configs
ARG GPU_ARCH=gfx1200
ENV FORCE_CUDA=1
ENV HCC_AMDGPU_TARGET=${GPU_ARCH}
ENV AMDGPU_TARGETS=${GPU_ARCH}
ENV PYTORCH_ROCM_ARCH=${GPU_ARCH}
ENV GPU_ARCHS=${GPU_ARCH}
ENV HIP_PLATFORM=amd
ENV ROCM_HOME=/opt/rocm
ENV CC=/opt/rocm/bin/hipcc
ENV CXX=/opt/rocm/bin/hipcc
ENV HSA_OVERRIDE_GFX_VERSION=12.0.0


# Run the installation and compilation script.
# This clones ComfyUI-Trellis2-GGUF, patches it, and builds all native extensions
# (CuMesh, FlexGEMM, o-voxel, nvdiffrast, and the GL plugin) from source.
RUN /app/install-trellis2-gguf-rocm.sh

# Install triton nightly and flash-attention for ROCm
RUN pip install --no-cache-dir --pre triton --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/
RUN git clone https://github.com/Dao-AILab/flash-attention /app/flash-attention && \
    cd /app/flash-attention && \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install --no-cache-dir --no-build-isolation .

# Patch aiter bug where it uses hipcc -v which tries to link on ROCm 7.0 and fails
RUN sed -i 's/\[compiler, "-v"\]/\[compiler, "--version"\]/g' /opt/venv/lib/python3.12/site-packages/aiter/jit/utils/cpp_extension.py
# Expose ComfyUI port
EXPOSE 8188

# Set the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
