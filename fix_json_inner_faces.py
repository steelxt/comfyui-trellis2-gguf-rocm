import json
import glob
import os

workspace_dir = "/home/steelx/Projects/comfy trellis amd/comfyui-trellis2-gguf-rocm"
json_files = glob.glob(os.path.join(workspace_dir, "*.json"))

for file_path in json_files:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        modified = False
        for node in data.get("nodes", []):
            if "ReconstructMeshWithQuad" in node.get("type", ""):
                wv = node.get("widgets_values", [])
                # The boolean is typically the last or 3rd/4th widget. Let's find it by type!
                # Actually, in ComfyUI it's just a boolean in the list.
                for i in range(len(wv)):
                    if isinstance(wv[i], bool) and wv[i] is False:
                        wv[i] = True
                        modified = True
                        print(f"Patched remove_inner_faces to True in {os.path.basename(file_path)} node ID {node.get('id')}")
        
        if modified:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
                
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
