import json

file_path = "/home/steelx/Documents/PixelArtistry_Pixal3D_LowPoly.json"
with open(file_path, "r") as f:
    data = json.load(f)

for node in data.get("nodes", []):
    if node.get("type") == "Trellis2MeshTexturing_GGUF":
        # Overwrite with the correct default widgets for the GGUF version
        node["widgets_values"] = [
            0,            # seed
            "fixed",      # control_after_generate
            12,           # texture_steps
            3.0,          # texture_guidance_strength
            0.20,         # texture_guidance_rescale
            3.0,          # texture_rescale_t
            "1024",       # resolution
            4096,         # texture_size
            "OPAQUE",     # texture_alpha_mode
            False,        # double_side_material
            0.0,          # texture_guidance_interval_start
            0.90,         # texture_guidance_interval_end
            4,            # max_views
            False,        # bake_on_vertices
            False,        # use_custom_normals
            "Xatlas",     # uv_unwrap_method
            60.0,         # mesh_cluster_threshold_cone_half_angle_rad
            False,        # use_tiled_encoder
            512,          # encoder_tile_size
            24,           # encoder_overlap
            False,        # use_tiled_decoder_for_texture
            120,          # decoder_tile_size
            48,           # decoder_overlap
            "euler"       # sampler
        ]
        print(f"Fixed widgets for node {node['id']}")

with open(file_path, "w") as f:
    json.dump(data, f, indent=2)

print("Texturing node successfully modified!")
