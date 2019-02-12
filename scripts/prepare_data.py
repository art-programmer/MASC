# Exports a train scan in the evaluation format using:
#   - the *_vh_clean_2.ply mesh
#   - the labels defined by the *.aggregation.json and *_vh_clean_2.0.010000.segs.json files
#
# example usage: export_train_mesh_for_evaluation.py --scan_path [path to scan data] --output_file [output file] --type label
# Note: technically does not need to load in the ply file, since the ScanNet annotations are defined against the mesh vertices, but we load it in here as an example.

# python imports
import math
import os, sys
import inspect
import json
import glob

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
from scripts.util import read_label_mapping
from scripts.util_3d import read_mesh_vertices
import torch
import multiprocessing as mp
import functools
#from utils import write_ply_label

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(filename, label_map):
    scan_name = filename.split('_vh')[0]
    mesh_file = os.path.join(scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(scan_name + '.aggregation.json')
    seg_file = os.path.join(scan_name + '_vh_clean_2.0.010000.segs.json')

    print(filename)        
    if os.path.exists(mesh_file[:-4] + '.pth'):
        return
    
    mesh_vertices, mesh_colors, faces = read_mesh_vertices(mesh_file)
    if os.path.exists(agg_file):
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        seg_to_verts, num_verts = read_segmentation(seg_file)
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)     # 0: unannotated
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated

        # write_ply_label('test/mesh.ply', mesh_vertices, faces, label_ids)
        # exit(1)
        
        for label, segs in label_to_segs.items():
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id

        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
    else:
        num_verts = len(mesh_vertices)
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)     # 0: unannotated
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        pass
    mesh_vertices = np.ascontiguousarray(mesh_vertices - mesh_vertices.mean(0))
    mesh_colors = np.ascontiguousarray(mesh_colors) / 127.5 - 1
    torch.save((mesh_vertices, mesh_colors, label_ids, instance_ids, faces), mesh_file[:-4] + '.pth')
    return

def prepare_data(options):
    ROOT_FOLDER = options.dataFolder
    files = sorted(glob.glob(options.dataFolder + '*/*_vh_clean_2.ply'))
    p = mp.Pool(processes=mp.cpu_count())

    label_map = read_label_mapping(options.labelFile, label_from='raw_category', label_to='nyu40id')

    p.map(functools.partial(export, label_map=label_map), files)
    p.close()
    p.join()
    return
