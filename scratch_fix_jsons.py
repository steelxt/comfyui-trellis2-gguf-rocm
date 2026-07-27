import json
import sys

file_path = "/home/steelx/Documents/PixelArtistry_Pixal3D_LowPoly.json"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading {file_path}: {e}")
    sys.exit(1)

fixed_count = 0
if "nodes" in data:
    for node in data["nodes"]:
        node_type = node.get("type", "")
        # If it's a Trellis2 node but doesn't have the _GGUF suffix, add it
        if node_type.startswith("Trellis2") and not node_type.endswith("_GGUF"):
            new_type = f"{node_type}_GGUF"
            print(f"Fixed Node {node.get('id')}: {node_type} -> {new_type}")
            node["type"] = new_type
            fixed_count += 1
            
        # Also check and fix the aux_id in properties just to be perfectly clean
        if "properties" in node and "aux_id" in node["properties"]:
            if node["properties"]["aux_id"] == "visualbruno/ComfyUI-Trellis2":
                node["properties"]["aux_id"] = "Aero-Ex/ComfyUI-Trellis2-GGUF"
                
        if "properties" in node and "Node name for S&R" in node["properties"]:
            s_and_r = node["properties"]["Node name for S&R"]
            if s_and_r.startswith("Trellis2") and not s_and_r.endswith("_GGUF"):
                node["properties"]["Node name for S&R"] = f"{s_and_r}_GGUF"

if fixed_count > 0:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSuccessfully fixed {fixed_count} nodes in {file_path}!")
else:
    print(f"\nNo nodes needed fixing in {file_path}. They might already have _GGUF!")
