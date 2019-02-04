# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob, plyfile, numpy as np, multiprocessing as mp, torch
import os

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x]=i

ROOT_FOLDER = '/gruvi/Data/chenliu/ScanNet/scans/'
files=sorted(glob.glob(ROOT_FOLDER + '*/*_vh_clean_2.ply'))
#files2=sorted(glob.glob(ROOT_FOLDER + '*/*_vh_clean_2.labels.ply'))
#assert len(files) == len(files2)

def f(fn):
    print(fn[:-4] + '.pth')    
    if os.path.exists(fn[:-4] + '.pth'):
        return
    #fn2 = fn[:-3]+'labels.ply'
    a=plyfile.PlyData().read(fn)
    v=np.array([list(x) for x in a.elements[0]])
    coords=np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0))
    colors=np.ascontiguousarray(v[:,3:6])/127.5-1
    #a=plyfile.PlyData().read(fn2)
    #w=remapper[np.array(a.elements[0]['label'])]

    #filename = ROOT_FOLDER + scene_id + '/' + scene_id + '.aggregation.json'
    data = json.load(open(fn.replace('vh_clean_2.ply', 'aggregation.json'), 'r'))
    aggregation = np.array(data['segGroups'])

    data = json.load(open(fn.replace('vh_clean_2.ply', 'vh_clean_2.0.010000.segs.json'), 'r'))
    segmentation = np.array(data['segIndices'])

    groupSegments = []
    groupLabels = []
    for segmentIndex in xrange(len(aggregation)):
        groupSegments.append(aggregation[segmentIndex]['segments'])
        groupLabels.append(aggregation[segmentIndex]['label'])
        continue

    segmentation = segmentation.astype(np.int32)    
    torch.save((coords,colors,w, instance),fn[:-4]+'.pth')
    
p = mp.Pool(processes=mp.cpu_count())
p.map(f,files)
p.close()
p.join()
