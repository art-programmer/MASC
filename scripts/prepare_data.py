# Exports a train scan in the evaluation format using:
#   - the *_vh_clean_2.ply mesh
#   - the labels defined by the *.aggregation.json and *_vh_clean_2.0.010000.segs.json files
#
# example usage: export_train_mesh_for_evaluation.py --scan_path [path to scan data] --output_file [output file] --type label
# Note: technically does not need to load in the ply file, since the ScanNet annotations are defined against the mesh vertices, but we load it in here as an example.

# python imports
import math
import os, sys, argparse
import inspect
import json
import glob

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import util
import util_3d
import torch
import multiprocessing as mp    
from utils import write_ply_label

TASK_TYPES = {'label', 'instance'}

parser = argparse.ArgumentParser()
#parser.add_argument('--scan_path', required=True, help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
parser.add_argument('--label_map_file', default='/gruvi/Data/chenliu/ScanNet/tasks/scannetv2-labels.combined.tsv', help='path to scannetv2-labels.combined.tsv')
parser.add_argument('--type', default='instance', help='task type [label or instance]')
opt = parser.parse_args()
assert opt.type in TASK_TYPES

label_map = util.read_label_mapping(opt.label_map_file, label_from='raw_category', label_to='nyu40id')
# remapper=np.ones(150)*(-100)
# for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
#     remapper[x]=i


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


def export(filename):
    scan_name = filename.split('_vh')[0]
    mesh_file = os.path.join(scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(scan_name + '.aggregation.json')
    seg_file = os.path.join(scan_name + '_vh_clean_2.0.010000.segs.json')

    if os.path.exists(mesh_file[:-4] + '.pth') and len(torch.load(mesh_file[:-4] + '.pth')) == 5 and False:
        return
    print(filename)    
    
    #mesh_vertices, mesh_colors, faces = util_3d.read_mesh_vertices(mesh_file)
    if os.path.exists(agg_file):
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        seg_to_verts, num_verts = read_segmentation(seg_file)
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)     # 0: unannotated
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated

        invalid_instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        # print(len(seg_to_verts))
        for object_id, segs in object_id_to_segs.items():
            object_verts = []
            for seg in segs:
                verts = seg_to_verts[seg]
                object_verts.append(verts)
                continue
            nums = np.array([len(verts) for verts in object_verts])
            invalid_indices = np.logical_and(nums < (0.5 * nums.sum()), nums >= 100)
            invalid_indices = invalid_indices.nonzero()[0]
            if len(invalid_indices) == 0:
                continue                
            seg = segs[np.random.choice(invalid_indices)]
            verts = seg_to_verts[seg]
            
            invalid_instance_ids[verts] = object_id            
            continue
        torch.save((invalid_instance_ids, ), mesh_file[:-4] + '_invalid.pth')
        return
    
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
    #point_cloud = torch.load(mesh_file[:-4] + '.pth')
    #print(point_cloud)
    #print([(v.shape, v.min(0), v.max(0), v.dtype) for v in point_cloud])
    #print(np.abs(remapper[label_ids] - point_cloud[2]).max())
    #print([(v.shape, v.min(), v.max()) for v in [mesh_vertices, mesh_colors, label_ids, instance_ids]])
    #exit(1)
    mesh_vertices = np.ascontiguousarray(mesh_vertices - mesh_vertices.mean(0))
    mesh_colors = np.ascontiguousarray(mesh_colors) / 127.5 - 1
    # print(np.abs(mesh_vertices - point_cloud[0]).max())
    # print(np.abs(mesh_colors - point_cloud[1]).max())            
    # print(np.abs(remapper[label_ids] - point_cloud[2]).max())
    # exit(1)    
    torch.save((mesh_vertices, mesh_colors, label_ids, instance_ids, faces), mesh_file[:-4] + '.pth')
    return

def main():
    ROOT_FOLDER = '/gruvi/Data/chenliu/ScanNet/scans/'
    files = sorted(glob.glob(ROOT_FOLDER + '*/*_vh_clean_2.ply'))
    #print(files)
    #exit(1)

    # files = [filename for filename in files if 'scene0568_00' in filename]
    # print(files)
    # export(files[0])
    # exit(1)

    
    #print(mp.cpu_count())
    # for filename in files:
    #     export(filename)
    #     continue
    # exit(1)
    p = mp.Pool(processes=mp.cpu_count())
    p.map(export, files)
    p.close()
    p.join()
    
    #for filename in files:
    #export(mesh_file, agg_file, seg_file, opt.label_map_file)


if __name__ == '__main__':
    main()
