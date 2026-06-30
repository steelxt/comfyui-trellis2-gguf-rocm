- [x] Create `entrypoint.sh` with runtime patches
- [x] Create `Dockerfile`
- [x] Create `docker-compose.yml`
- [x] Verify Dockerfile building
- [x] Fix double-patching syntax errors on startup in `entrypoint.sh` and `install-trellis2-gguf-rocm.sh`
- [x] Launch container and verify custom nodes import successfully without errors
- [x] Patch native `ComfyUI-GGUF/loader.py` to clone GGUF mapped weights on CPU
- [x] Patch `trellis2_image_to_3d.py` to call `torch.cuda.synchronize()` before unloading models
- [x] Restart container and verify patches are applied
- [ ] Verify successful completion of the full generation pipeline without segfaults




