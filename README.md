# ComfyUI Trellis 2 GGUF — AMD ROCm & RDNA4 Edition

Trellis2 GGUF custom nodes for ComfyUI, heavily patched and optimized to build and run natively on **AMD GPUs via ROCm (including RDNA4 architecture)**.

Inspired by https://www.youtube.com/watch?v=FuFm8zBHDWI.  
Started from the Windows installer at https://pixel-artistry.com/trellis2gguf and adapted to Linux + ROCm 7.2 with full Flash Attention and CuMesh support.

---

## 🚀 Tested Environment & Specs

| Component | Version / Description |
|-----------|-----------------------|
| **GPU** | **AMD Radeon RX 9060 XT (gfx1200 / RDNA4)** |
| **CPU** | **AMD Ryzen 9 9950X3D (32-core) @ 5.76 GHz** |
| **OS** | Arch Linux / Linux (Kernel 6.x) |
| **ROCm** | 7.2 |
| **Python** | 3.12 / 3.14 |
| **PyTorch** | 2.11.0+rocm7.2 (Nightly / Custom ROCm build) |

---

## 🛠️ Critical Compilation & Execution Details

To successfully compile and execute Trellis 2 3D generation on modern AMD ROCm cards (specifically RDNA3 `gfx1100`/`gfx1102` and RDNA4 `gfx1200`), several key modifications and compiler patches are applied in this project:

### 1. Architecture Targeting (`gfx1200`)
- When compiling C++ and HIP kernels (CuMesh, nvdiffrast, o-voxel), the target GPU architecture must be explicitly defined to avoid generic fallbacks or instruction incompatibilities.
- The Dockerfile and environment scripts export `PYTORCH_ROCM_ARCH="gfx1200"` and set `TORCH_ROCM_AARCH64=0`.

### 2. Native Flash Attention for AMD ROCm
- Running unquantized attention on 3D voxel slats at high resolutions (e.g., 1024 or 4096 texture size) requires massive VRAM bandwidth and will quickly throw Out-Of-Memory (OOM) exceptions on standard SDPA backends.
- We compile `Dao-AILab/flash-attention` directly from source with AMD Triton support enabled (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`).
- To prevent CPU RAM / Swap OOM freezes during compilation, `MAX_JOBS=4` is enforced in the build pipeline.

### 3. Triton Pre-Release Requirement
- Compiling Flash Attention kernels for RDNA4 (`gfx1200`) requires bleeding-edge Triton compiler features. The build script automatically injects `pip install --no-cache-dir --pre triton` prior to compiling Flash Attention.

### 4. Aiter Compiler Linker Patch
- When compiling Flash Attention against ROCm 7.0+, the upstream `aiter` setup script crashes due to passing `-v` to the HIP compiler during version verification.
- Our Docker build automatically runs a `sed` patch on the cloned Flash Attention repository:
  ```bash
  sed -i 's/\[compiler, "-v"\]/\[compiler, "--version"\]/g' setup.py
  ```

### 5. Runtime Sliced & Windowed Attention Fallbacks
- On boot, `entrypoint.sh` automatically patches Trellis2 attention operators in-memory to utilize sparse windowed attention and dynamic memory slicing if VRAM limits are approached during 4K texture decoding.

---

## 🐳 Docker Setup (Recommended)

To run in an isolated container with all HIP kernels, OpenGL EGL headless bindings, and Flash Attention pre-compiled:

### 1. Build and Launch
```bash
docker compose up -d --build
```

### 2. Auto-Installed ComfyUI Manager
- The container's startup script (`entrypoint.sh`) checks for `ComfyUI-Manager` on boot. If missing from your mounted host volume, it automatically clones and initializes the Manager without requiring a Docker image rebuild.

### 3. Permission & Host Volume Management
Because Docker runs as `root` internally, mounted volumes (`./models`, `./output`, `./input`, `./user`) can become locked by root ownership on the Linux host. To grant full control back to your local user (e.g., `steelx`) along with other users:
```bash
# Reassign volume ownership from inside the container without sudo:
docker exec -u 0 comfyui-trellis bash -c "chown -R 1000:1000 /app/ComfyUI/models /app/ComfyUI/output /app/ComfyUI/input /app/ComfyUI/user"

# Grant global Read/Write/Execute permissions across the project:
chmod -R a+rwX .
```

---

## 📦 What the Install Script Patches (C++ / HIP Kernels)

The original Trellis2 GGUF nodes depend on CUDA-only C++ extensions. The build script (`install-trellis2-gguf-rocm.sh`) automatically applies the following ROCm translations at compile time:

### nvdiffrast
- Replaces `__frcp_rz` (CUDA intrinsic) with `__fdividef`.
- Casts warp sync masks to 64-bit (a strict ROCm 7.x requirement).
- Removes the `cudaraster` module (NVIDIA PTX assembly) and provides runtime stubs. The **OpenGL rasterizer** (`RasterizeGLContext`) handles all 3D mesh projection.
- Renames `.cpp` → `.cu` so `hipcc` compiles files requiring CUDA→HIP header translation.

#### Headless EGL Plugin (CPU-Bounce Path)
The GL rasterizer plugin is rebuilt with Mesa EGL headless surfaceless context creation (removing X11/GLX dependencies):
- **Upload** (GPU→GL): `hipMemcpy D2H` → `glBufferSubData`
- **Readback** (GL→GPU): `glGetTexImage` → `hipMemcpy H2D`
- Links `GL`, `EGL`, `amdhip64`.

### nvdiffrec_render
- Removes `-lcuda -lnvrtc` linker flags and patches CUDA headers to HIP equivalents.

### CuMesh
- Replaces `::cuda::std::tuple` with `rocprim::tuple`.
- Fixes brace-init for explicit rocprim constructors and removes NVCC-only compiler flags.

---

## 🧩 Workflow & JSON Compatibility Notes

### 1. Node Naming (`_GGUF` Suffix)
The GGUF variant uses node names with a `_GGUF` suffix (e.g., `Trellis2SimplifyMesh_GGUF`). If loading workflows built for the original (non-GGUF) Trellis2 plugin by creators like *visualbruno* or *pixel artistry*, you must append `_GGUF` to the node type names in your JSON file.

### 2. All-In-One Advanced Generator vs. Granular Nodes
- The original Trellis workflows often break down generation into 4 separate nodes: `ImageCondGenerator` → `SparseGenerator` → `ShapeGenerator` → `DecodeLatents`.
- **Recommended**: Delete those 4 granular nodes in ComfyUI and replace them with a single **`Trellis2 - Advanced Mesh with Voxel Generator (GGUF)`** node. It wraps the entire pipeline, runs faster, and handles VRAM offloading automatically.

### 3. Avoiding Tiling Crashes on Texture Decoders
- When importing older JSON workflows, widget indices can misalign with new GGUF parameters. If `use_tiled_decoder_for_texture` gets accidentally mapped to `True`, generation will crash with:
  ```text
  AttributeError: 'SparseUnetVaeDecoder' object has no attribute '_tiled_forward'
  ```
- **Fix**: The texture VAE decoder does not support tiled forwarding. Always ensure `use_tiled_decoder_for_texture` is set to **`False`** on `Trellis2MeshTexturing_GGUF` nodes. (Shape decoders support tiling without issue).

---

## 🖥️ Manual / Host Setup (Non-Docker)

If running directly on Arch Linux without Docker:

```bash
# 1. Install ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install comfy-cli
comfy install --restore

# 2. Compile and install patched Trellis2 GGUF nodes
./install-trellis2-gguf-rocm.sh

# 3. Launch with ROCm optimizations
./run.sh
```
`run.sh` automatically sets `HSA_OVERRIDE_GFX_VERSION` and `ATTN_BACKEND=sdpa` before booting ComfyUI.