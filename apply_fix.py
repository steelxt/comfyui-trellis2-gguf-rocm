import json

file_path = "/home/steelx/Documents/PixelArtistry_Pixal3D_LowPoly.json"
with open(file_path, "r") as f:
    data = json.load(f)

nodes = data.get("nodes", [])
links = data.get("links", [])

to_remove_ids = {217, 213, 214, 215}

# Remove old nodes
new_nodes = [n for n in nodes if n["id"] not in to_remove_ids]

# Remove links completely internal to the removed nodes, and the old incoming links
new_links = []
for l in links:
    link_id, from_id, from_port, to_id, to_port, link_type = l
    if to_id in to_remove_ids or from_id in to_remove_ids:
        continue
    new_links.append(l)

# Create Node 999
node_999 = {
  "id": 999,
  "type": "Trellis2MeshWithVoxelAdvancedGenerator_GGUF",
  "pos": [-1142, 222],
  "size": [400, 600],
  "flags": {},
  "order": 15,
  "mode": 0,
  "inputs": [
    {
      "name": "pipeline",
      "type": "TRELLIS2PIPELINE",
      "link": 535
    },
    {
      "name": "image",
      "type": "IMAGE",
      "link": 394
    }
  ],
  "outputs": [
    {
      "name": "mesh",
      "type": "MESHWITHVOXEL",
      "links": [397]
    }
  ],
  "properties": {
    "Node name for S&R": "Trellis2MeshWithVoxelAdvancedGenerator_GGUF"
  },
  "widgets_values": [
    12348,          # seed
    "1024_cascade", # pipeline_type
    12,             # sparse_structure_steps
    7.5,            # sparse_structure_guidance_strength
    0.05,           # sparse_structure_guidance_rescale
    4.0,            # sparse_structure_rescale_t
    12,             # shape_steps
    7.5,            # shape_guidance_strength
    0.05,           # shape_guidance_rescale
    4.0,            # shape_rescale_t
    12,             # texture_steps
    3.0,            # texture_guidance_strength
    0.20,           # texture_guidance_rescale
    3.0,            # texture_rescale_t
    49152,          # max_num_tokens
    4,              # max_views
    32,             # sparse_structure_resolution
    True,           # generate_texture_slat
    0.1,            # sparse_structure_guidance_interval_start
    1.0,            # sparse_structure_guidance_interval_end
    0.1,            # shape_guidance_interval_start
    1.0,            # shape_guidance_interval_end
    0.0,            # texture_guidance_interval_start
    0.9,            # texture_guidance_interval_end
    True,           # use_tiled_decoder
    "heun",         # sparse_structure_sampler
    "heun",         # shape_sampler
    "euler"         # texture_sampler
  ]
}
new_nodes.append(node_999)

# Re-add the essential links pointing to/from Node 999
new_links.append([535, 282, 0, 999, 0, "TRELLIS2PIPELINE"])
new_links.append([394, 194, 0, 999, 1, "IMAGE"])
new_links.append([397, 999, 0, 193, 0, "MESHWITHVOXEL"])

data["nodes"] = new_nodes
data["links"] = new_links

with open(file_path, "w") as f:
    json.dump(data, f, indent=2)

print("Workflow successfully modified!")
