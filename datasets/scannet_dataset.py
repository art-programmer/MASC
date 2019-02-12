from torch.utils.data import Dataset
import torch
import numpy as np
import time
import os
import scipy.ndimage
from utils import remapper

blur0=np.ones((3,1,1)).astype('float32')/3
blur1=np.ones((1,3,1)).astype('float32')/3
blur2=np.ones((1,1,3)).astype('float32')/3

def elastic(x,gran,mag):
    bb=np.abs(x).max(0).astype(np.int32)//gran+3
    noise=[np.random.randn(bb[0],bb[1],bb[2]).astype('float32') for _ in range(3)]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur0,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur1,mode='constant',cval=0) for n in noise]
    noise=[scipy.ndimage.filters.convolve(n,blur2,mode='constant',cval=0) for n in noise]
    ax=[np.linspace(-(b-1)*gran,(b-1)*gran,b) for b in bb]

    interp=[scipy.interpolate.RegularGridInterpolator(ax,n,bounds_error=0,fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    noise = g(x)
    return x+g(x)*mag

## ScanNet dataset class
class ScanNetDataset(Dataset):
    def __init__(self, options, split, load_confidence=False, random=True):
        self.options = options
        self.split = split
        self.random = random
        self.imagePaths = []
        self.dataFolder = options.dataFolder
        self.load_confidence = load_confidence
        
        with open('split_' + split + '.txt', 'r') as f:
            for line in f:
                scene_id = line.strip()
                if len(scene_id) < 5 or scene_id[:5] != 'scene':
                    continue
                if options.scene_id != '' and options.scene_id not in scene_id:
                    continue
                filename = self.dataFolder + '/' + scene_id + '/' + scene_id + '_vh_clean_2.pth'
                if os.path.exists(filename):
                    info = torch.load(filename)
                    if len(info) == 5:
                        self.imagePaths.append(filename)

                        #np.savetxt('semantic_val/' + scene_id + '.txt', info[2], fmt='%d')
                        pass
                    pass
                if split != 'train' and len(self.imagePaths) >= options.numTestingImages:
                    break
                continue
            pass
        
        #self.imagePaths = [filename for filename in self.imagePaths if 'scene0217_00' in filename]

        if options.numTrainingImages > 0 and split == 'train':
            self.numImages = options.numTrainingImages
        else:
            self.numImages = len(self.imagePaths)            
            pass
        return
    
    def __len__(self):
        return self.numImages

    def __getitem__(self, index):
        if self.random:
            t = int(time.time() * 1000000)
            np.random.seed(((t & 0xff000000) >> 24) +
                           ((t & 0x00ff0000) >> 8) +
                           ((t & 0x0000ff00) << 8) +
                           ((t & 0x000000ff) << 24))
            index = np.random.randint(len(self.imagePaths))
        else:
            index = index % len(self.imagePaths)
            pass

        debug = -1
        if debug >= 0:
            index = debug
            print(index, self.imagePaths[index])
            pass

        coords, colors, labels, instances, faces = torch.load(self.imagePaths[index])
        invalid_instances, = torch.load(self.imagePaths[index].replace('.pth', '_invalid.pth'))
                                       
        labels = remapper[labels]
        
        #neighbor_gt = torch.load(self.imagePaths[index].replace('.pth', '_neighbor.pth'))
        #print(neighbor_gt[0])
        #exit(1)
        #neighbor_gt = 1
        #print(coords.min(0), coords.max(0))
        if self.split == 'train':
            m = np.eye(3) + np.random.randn(3,3) * 0.1
            m[0][0] *= np.random.randint(2) * 2 - 1
            theta = np.random.rand() * 2 * np.pi
        else:
            m = np.eye(3)
            theta = 0
            pass
        
        scale = self.options.scanScale
        full_scale = self.options.inputScale
        m *= scale
        m = np.matmul(m, [[np.cos(theta), np.sin(theta),0], [-np.sin(theta), np.cos(theta),0], [0,0,1]])
        coords = np.matmul(coords, m)
        if self.split == 'train':
            coords = elastic(coords, 6 * scale // 50,40 * scale / 50)
            #coords = elastic(coords, 20 * scale // 50, 160 * scale / 50)
            pass

        if 'normal' in self.options.suffix:
            points_1 = coords[faces[:, 0]]
            points_2 = coords[faces[:, 1]]
            points_3 = coords[faces[:, 2]]
            face_normals = np.cross(points_2 - points_1, points_3 - points_1)
            face_normals /= np.maximum(np.linalg.norm(face_normals, axis=-1, keepdims=True), 1e-4)
            normals = np.zeros((len(coords), 3))
            for c in range(3):
                np.add.at(normals, faces[:, c], face_normals)
                continue
            normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-4)
            colors = np.concatenate([colors, normals], axis=-1)
            pass

        if self.split == 'train':
            colors[:, :3] = colors[:, :3] + np.random.randn(3) * 0.1            
            pass
        
        if self.load_confidence:
            scene_id = self.imagePaths[index].split('/')[-1].split('_vh_clean_2')[0]
            info = torch.load('test/output_normal_augment_2_' + self.split + '/cache/' + scene_id + '.pth')
            if len(info) == 2:
                semantic_pred, instance_pred = info
            else:
                semantic_pred, instance_pred = info[3], info[6]
                semantic_pred = semantic_pred[:len(coords)]                                
                instance_pred = instance_pred[:len(coords)]                
                pass            
            instance_pred += 1
            unique_instances, indices, counts = np.unique(instances, return_index=True, return_counts=True)
            
            instance_counts = np.zeros(unique_instances.max() + 1)
            instance_counts[unique_instances] = counts
            instance_semantics = np.zeros(unique_instances.max() + 1)
            instance_semantics[unique_instances] = labels[indices]
            confidence_gt = []
            semantic_gt = []
            instance_masks = []
            new_coords = np.zeros(coords.shape, dtype=coords.dtype)            
            for instance in range(instance_pred.max() + 1):
                instance_mask = instance_pred == instance
                if instance_mask.sum() == 0:
                    print('sum = 0', instance, instance_pred.max() + 1, instance_mask.sum())
                    exit(1)
                info = np.unique(semantic_pred[instance_mask > 0.5], return_counts=True)
                label_pred = info[0][info[1].argmax()]
                info = np.unique(instances[instance_mask > 0.5], return_counts=True)
                instance_gt = info[0][info[1].argmax()]

                instance_coords = coords[instance_mask]
                mins = instance_coords.min(0)
                maxs = instance_coords.max(0)
                max_range = (maxs - mins).max()
                if self.split == 'train':
                    padding = (maxs - mins) * np.random.random(3) * 0.1
                else:
                    padding = max_range * 0.05
                    pass
                max_range += padding * 2
                mins = (mins + maxs) / 2 - max_range / 2
                instance_coords = np.clip(np.round((instance_coords - mins) / max_range * full_scale), 0, full_scale - 1)
                new_coords[instance_mask] = instance_coords

                if instance > 0:
                    confidence_gt.append(int(label_pred == instance_semantics[instance_gt] and info[1].max() > 0.5 * instance_counts[instance_gt]))
                    semantic_gt.append(label_pred)
                    instance_masks.append(instance_mask)
                    pass
                continue
            coords = np.concatenate([new_coords, np.expand_dims(instance_pred, -1)], axis=-1)
            sample = [coords.astype(np.int64), colors.astype(np.float32), faces.astype(np.int64), np.stack(semantic_gt).astype(np.int64), np.stack(confidence_gt).astype(np.int64), self.imagePaths[index], np.stack(instance_masks).astype(np.int32)]
            return sample        

        
        mins = coords.min(0)
        maxs = coords.max(0)
        #ranges = maxs - mins
        if self.split == 'train':
            offset = -mins + np.clip(full_scale - maxs + mins - 0.001, 0, None) * np.random.rand(3) + np.clip(full_scale - maxs + mins + 0.001, None, 0) * np.random.rand(3)
            coords += offset
        else:
            coords -= (mins + maxs) // 2 - full_scale // 2
            #coords -= mins
            pass
        
        coords = np.round(coords)
        if False:
            idxs = (coords.min(1) >= 0) * (coords.max(1) < full_scale)
            coords = coords[idxs]
            colors = colors[idxs]
            labels = labels[idxs]
            instances = instances[idxs]
            invalid_instances = invalid_instances[idxs]
        else:
            #print(coords.min(0), coords.max(0))
            #exit(1)
            coords = np.clip(coords, 0, full_scale - 1)
            pass
        
        coords = np.concatenate([coords, np.full((coords.shape[0], 1), fill_value=index)], axis=-1)
        #coords = np.concatenate([coords, np.expand_dims(instances, -1)], axis=-1)
        sample = [coords.astype(np.int64), colors.astype(np.float32), faces.astype(np.int64), labels.astype(np.int64), instances.astype(np.int64), invalid_instances.astype(np.int64), self.imagePaths[index]]
        return sample
