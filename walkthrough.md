# Walkthrough — Docker Deployment of ComfyUI Trellis2 GGUF for ROCm (gfx1200)

We have successfully created a clean Docker deployment setup for the `comfyui-trellis2-gguf-rocm` repository on AMD GPUs (targeting `gfx1200`, RDNA 4).

## Changes Made

1. **[install-trellis2-gguf-rocm.sh](file:///home/steelx/Projects/comfy%20trellis%20amd/comfyui-trellis2-gguf-rocm/install-trellis2-gguf-rocm.sh)**:
   - Patched to allow overriding `HCC_AMDGPU_TARGET`, `AMDGPU_TARGETS`, and `PYTORCH_ROCM_ARCH` build-time architectures via environment variables (falling back to `gfx1102` default).
2. **[entrypoint.sh](file:///home/steelx/Projects/comfy%20trellis%20amd/comfyui-trellis2-gguf-rocm/entrypoint.sh)**:
   - Checks if model storage folders exist.
   - Automatically downloads the required DINOv3 preprocessor model on startup if missing.
   - Automatically downloads the Trellis2 GGUF pipeline and model files on startup if missing.
   - Sets optimized ROCm performance variables (`ATTN_BACKEND=sdpa` and Triton variables) and starts ComfyUI with cross-attention enabled.
3. **[Dockerfile](file:///home/steelx/Projects/comfy%20trellis%20amd/comfyui-trellis2-gguf-rocm/Dockerfile)**:
   - Uses `rocm/pytorch:latest` as base.
   - Installs OpenGL/EGL/Mesa libraries needed to compile `nvdiffrast` CPU-bounce rasterizer, plus X11/GLib libraries (for OpenCV/Open3D GUI components) and `libsparsehash-dev`.
   - Forces standard C++ compilers to use ROCm's `hipcc` by setting `CC` and `CXX` flags.
   - Configures the build target parameters (`FORCE_CUDA=1`, `HCC_AMDGPU_TARGET`, `GPU_ARCHS`, etc.) and sets `HSA_OVERRIDE_GFX_VERSION=12.0.0` for runtime compatibility on your RX 9060 XT.
   - Sets the target build architecture to `gfx1200` by default.
   - Automatically runs the installation and compiles all C++ extensions during image build time.
4. **[docker-compose.yml](file:///home/steelx/Projects/comfy%20trellis%20amd/comfyui-trellis2-gguf-rocm/docker-compose.yml)**:
   - Configures `/dev/kfd` and `/dev/dri` GPU pass-through.
   - Maps video/render group access.
   - Configures volume mounts for models, inputs, outputs, and user state.
   - Pins execution to the RX 9060 XT GPU using `HIP_VISIBLE_DEVICES=0` to prevent collisions with CPU integrated graphics.

---

## Instructions for Deploying

To build and launch the container on CachyOS / Arch Linux:

### 1. Build the Docker Image
Run this command to build the image (which will compile all C++ packages specifically for your `gfx1200` GPU):
```bash
docker compose build
```

> [!NOTE]
> This command will download the base image and compile native C++ modules. It may take 10-20 minutes depending on internet speed and hardware performance.

### 2. Launch ComfyUI
Start the container in detached mode (background) or foreground:
```bash
docker compose up
```

- When running for the first time, the container will detect if you have downloaded the models.
- If they are missing, it will automatically download the DINOv3 preprocessor and the Trellis2 GGUF models (several gigabytes) to your local `./models` directory.
- Once downloaded, it will start ComfyUI.

### 3. Open ComfyUI
Open your browser and navigate to:
[http://localhost:8188](http://localhost:8188)

---

## Verification Results

The container build and execution have been successfully verified:

1. **Successful Native Compilation**: The Docker image successfully built and compiled all C++ native extensions (`CuMesh`, `FlexGEMM`, `o-voxel`, `nvdiffrast`, and `nvdiffrast_gl`) targeting the `gfx1200` GPU architecture.
2. **Double-Inclusion Fix**: Resolved the `redefinition of default argument` compilation error in the `nvdiffrast` build phase by introducing a preprocessor macro guard (`NVDR_FRAMEWORK_H_GUARD`) in the generated `framework.h` patch.
3. **Model Download & Persistence**: Verified that DINOv3 and all Trellis2 GGUF models were successfully downloaded directly into the host-mounted `./models` directory (preserving bandwidth and disk usage on restarts/rebuilds).
4. **ComfyUI Startup & Node Import**: ComfyUI successfully launched, recognized the host's discrete RX 9060 XT GPU on ROCm 7.0, and imported the custom node package without errors:
   ```text
   comfyui-trellis  | [INFO] Device: cuda:0 AMD Radeon RX 9060 XT : native
   comfyui-trellis  | [INFO] AMD arch: gfx1200
   comfyui-trellis  | [INFO] ROCm version: (7, 0)
   ...
   comfyui-trellis  | [INFO]    0.5 seconds: /app/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF
   ...
   comfyui-trellis  | [INFO] Starting server
   comfyui-trellis  | [INFO] To see the GUI go to: http://0.0.0.0:8188
   ```
5. **GGUF Loader & State Dict Fix**: Resolved the `SparseStructureFlowModel` loading mismatch error (`The size of tensor a (1536) must match the size of tensor b (864)`) by adding the `ComfyUI-GGUF` dependency custom node to the container's custom nodes. This activates the native GGUF tensor-handling modules (ops/dequant/loader) which directly assign quantized tensors and bypass PyTorch's default copy constraints, resulting in fast GPU-accelerated inference.
