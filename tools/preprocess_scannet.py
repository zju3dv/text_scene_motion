from tqdm import tqdm
import json
import numpy as np
import trimesh
from pathlib import Path

scannet_root = Path('data/ScanNet/scans')
# set the output folder
preprocess_folder = Path('data/scannet_preprocess')
preprocess_folder.mkdir(exist_ok=True, parents=True)

for folder in tqdm(scannet_root.iterdir()):
    npy_file = preprocess_folder / folder.stem
    
    ply2_file = list(folder.glob("*_vh_clean_2.ply"))[0]
    aggr_file= list(folder.glob("*_vh_clean.aggregation.json"))[0]
    segs_file = list(folder.glob("*_vh_clean_2*segs.json"))[0]
    mesh2 = trimesh.load(ply2_file)
    with open(aggr_file, 'r') as f:
        aggr_data = json.load(f)
    with open(segs_file, 'r') as f:
        segs_data = json.load(f)
    
    segIndices = np.array(segs_data['segIndices'])
    if len(segIndices) != len(mesh2.vertices):
        print(npy_file)
        print(len(segIndices))
        print(len(mesh2.vertices))
        segIndices = segIndices[:len(mesh2.vertices)]
    
    V = mesh2.vertices
    object_mask = np.zeros((len(V))) - 1
    for raw_aggr in aggr_data['segGroups']:
        segments = raw_aggr['segments']
        objectId = raw_aggr['objectId']
        for seg in segments:
            idx = (seg == segIndices)
            object_mask[idx] = int(objectId)
            
    scene_data = np.concatenate([
        mesh2.vertices,
        mesh2.visual.vertex_colors[:, :3],
        mesh2.vertex_normals,
        object_mask.reshape(-1, 1),
    ], axis=-1)
    
    np.save(npy_file, scene_data)