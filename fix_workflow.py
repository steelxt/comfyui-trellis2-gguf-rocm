import json

file_path = "/home/steelx/Documents/PixelArtistry_Pixal3D_LowPoly.json"
with open(file_path, "r") as f:
    data = json.load(f)

nodes = data.get("nodes", [])
links = data.get("links", [])

# Find the nodes to remove
to_remove_ids = set()
for n in nodes:
    if n["type"] in ["Trellis2ImageCondGenerator_GGUF", "Trellis2SparseGenerator_GGUF", "Trellis2ShapeGenerator_GGUF", "Trellis2DecodeLatents_GGUF"]:
        to_remove_ids.add(n["id"])

print("Nodes to remove:", to_remove_ids)

# Find what pipeline and image were going into these nodes
pipeline_link = None
image_link = None
for l in links:
    link_id, from_id, from_port, to_id, to_port, link_type = l
    if to_id in to_remove_ids:
        if link_type == "TRELLIS2PIPELINE":
            pipeline_link = l
        if link_type == "IMAGE":
            image_link = l

# Find where the mesh was going OUT from DecodeLatents
mesh_out_links = []
for l in links:
    link_id, from_id, from_port, to_id, to_port, link_type = l
    if from_id in to_remove_ids and to_id not in to_remove_ids:
        if link_type == "MESHWITHVOXEL":
            mesh_out_links.append(l)

print("Pipeline in:", pipeline_link)
print("Image in:", image_link)
print("Mesh out:", mesh_out_links)
