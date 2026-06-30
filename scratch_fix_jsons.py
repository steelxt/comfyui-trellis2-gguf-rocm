import json
import glob
import os

for f in glob.glob('/home/steelx/Projects/comfy trellis amd/comfyui-trellis2-gguf-rocm/*.json'):
    with open(f, 'r') as fp:
        try:
            data = json.load(fp)
        except:
            continue
    modified = False
    if 'nodes' in data:
        for node in data['nodes']:
            if node.get('type') in ('Trellis2ReconstructMeshWithQuad_GGUF', 'Trellis2ReconstructMesh_GGUF', 'Trellis2ReconstructMesh', 'Trellis2ReconstructMeshWithQuad'):
                if 'widgets_values' in node:
                    # In ReconstructMeshWithQuad, inputs are: remesh_band, resolution, remove_floaters, remove_inner_faces
                    # In ReconstructMesh, inputs are: resolution, remove_floaters, remove_inner_faces
                    
                    # We just know it's the last boolean in the list or the second boolean
                    for i in range(len(node['widgets_values'])):
                        if isinstance(node['widgets_values'][i], bool) and node['widgets_values'][i] == False:
                            # It's likely remove_inner_faces
                            node['widgets_values'][i] = True
                            modified = True
                            
    if modified:
        with open(f, 'w') as fp:
            json.dump(data, fp, indent=2)
        print(f"Updated {f}")
