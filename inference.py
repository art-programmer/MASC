from functools import partial
import multiprocessing as mp
import sys
import torch
import numpy as np
import glob

from options import parse_args
from utils import *
from models.instance import Model, CoordAugmentation, NeighborGT
from train_confidence import InstanceValidator


def inference(filename, model, neighbor_model, augmentation_model, test_dir, augment=True, scan_scale=50, full_scale=4096, evaluate_loss=False, use_normal=False):
    """ Predict semantics and affinities """

    coords, colors, labels, instances, faces = torch.load(filename)
    m = np.eye(3) * scan_scale
    theta = 0

    m = np.matmul(m, [[np.cos(theta), np.sin(theta),0], [-np.sin(theta), np.cos(theta),0], [0,0,1]])
    coords = np.matmul(coords, m)
    if use_normal:
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
    
    mins = coords.min(0)
    maxs = coords.max(0)
    coords -= (mins + maxs) // 2 - full_scale // 2

    coords = np.round(coords)
    coords = np.clip(coords, 0, full_scale - 1)
    coords = np.concatenate([coords, np.full((len(coords), 1), fill_value=0)], axis=-1)
    labels = remapper[labels]
    
    coords, colors, faces, semantic_gt, instance_gt = torch.from_numpy(coords.astype(np.int64)).cuda(), torch.from_numpy(colors.astype(np.float32)).cuda(), torch.from_numpy(faces.astype(np.int64)).cuda(), torch.from_numpy(labels.astype(np.int64)).cuda(), torch.from_numpy(instances.astype(np.int64)).cuda()

    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    num_ori_coords = len(coords)
    if augment:
        augmented_coords, augmented_colors, augmented_instances, augmented_edges = augmentation_model(coords, faces, colors, instance_gt.unsqueeze(-1))
        coords = torch.cat([coords, augmented_coords], dim=0)
        colors = torch.cat([colors, augmented_colors], dim=0)
        instance_gt = torch.cat([instance_gt, augmented_instances.squeeze(-1)], dim=0)
        edges = torch.cat([edges, augmented_edges], dim=0)
        pass
    
    semantic_pred, neighbor_pred = model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])))
    #semantic_pred, neighbor_pred = semantic_pred[0], neighbor_pred[0]

    if evaluate_loss:    
        semantic_loss = torch.nn.functional.cross_entropy(semantic_pred[:num_ori_coords].view((-1, int(semantic_pred.shape[-1]))), semantic_gt.view(-1))
        pass
    semantic_pred = semantic_pred.max(-1)[1]

    for neighbor in neighbor_pred:
        neighbor.features = torch.sigmoid(neighbor.features)
        continue

    if evaluate_loss:
        neighbor_gt = neighbor_model(coords.reshape((-1, 4)), instance_gt.reshape((-1, )))

        neighbor_losses = []
        for scale in range(len(neighbor_gt)):
            pred = neighbor_pred[scale].features
            gt = neighbor_gt[scale].features[:, :pred.shape[-1]]
            mask = neighbor_gt[scale].features[:, pred.shape[-1]:]
            neighbor_losses.append(torch.sum(torch.nn.functional.binary_cross_entropy(pred, (gt > 0.5).float(), weight=(1 - gt) * 9 + 1, reduce=False) * mask) / torch.clamp(mask.sum(), min=1))
            continue
        pass


    if evaluate_loss:
        losses = [semantic_loss] + neighbor_losses
        loss = sum(losses)

        loss_values = [l.data.item() for l in losses]
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f '%l
            continue
        #sys.stdout.write('\r' + status)
        print(status)
        # print(coords.min(0)[0], coords.max(0)[0], colors.min(), colors.max())
        # print(np.unique(semantic_pred.detach().cpu().numpy()), (neighbor_pred[0].features < 0.5).sum(), neighbor_pred[0].features.shape, neighbor_pred[0].features.min(), neighbor_pred[0].features.max())
        # exit(1)
        pass

    scene_id = filename.split('/')[-1].split('_vh_clean')[0]
    
    semantic_pred = semantic_pred.detach().cpu().numpy()
    semantic_gt = semantic_gt.detach().cpu().numpy()
    semantic_gt[semantic_gt == -100] = 20
    instance_gt = instance_gt.detach().cpu().numpy()
    neighbors = neighbor_model.toDense(neighbor_pred)
    neighbors = [neighbor.detach().cpu().numpy() for neighbor in neighbors]
    coords = coords.detach().cpu().numpy()
    coords = coords[:, :3]
    colors = colors.detach().cpu().numpy()
    colors = np.clip((colors + 1) * 127.5, 0, 255).astype(np.uint8)    
    edges = edges.detach().cpu().numpy()    
    if True:
        writeSemantics(test_dir + '/sem_pred/' + scene_id + '.txt', semantic_pred)
        #writeSemantics(test_dir + '/sem_gt/' + str(sample_index * options.batchSize + batch_index) + '.txt', semantic_gt[batch_index])
        pass

    torch.save((coords, colors, edges, semantic_pred, neighbors, num_ori_coords), test_dir + '/cache/' + scene_id + '.pth')
    return


def group(filename, test_dir, num_scales, augment=True, num_cross_scales=0, scan_scale=50, full_scale=4096):
    """ Run the clustering algorithm """

    scene_id = filename.split('/')[-1].split('_vh_clean')[0]    
    coords, colors, edges, semantic_pred, neighbors, num_ori_coords = torch.load(test_dir + '/cache/' + scene_id + '.pth')
    #print(coords.min(0), coords.max(0), np.unique(semantic_pred), neighbors[0].min(), neighbors[0].max())
    instance_pred, _ = findInstances(coords, edges, semantic_pred, neighbors, num_scales, num_cross_scales, print_info=True)

    if augment:
        instance_labels, counts = np.unique(instance_pred[:num_ori_coords], return_counts=True)
        valid_labels = instance_labels[np.logical_and(counts > 100, instance_labels >= 0)]        
        #print('num valid instances', len(valid_labels))
        label_map = np.full(instance_pred.max() + 1, fill_value=-1, dtype=np.int32)
        for index, label in enumerate(valid_labels):
            label_map[label] = index
            continue
        instance_pred = label_map[instance_pred]
        pass

    if 'train' in test_dir:
        torch.save((semantic_pred[:num_ori_coords], instance_pred[:num_ori_coords]), test_dir + '/cache/' + scene_id + '.pth')
    else:
        torch.save((coords, colors, edges, semantic_pred, neighbors, num_ori_coords, instance_pred), test_dir + '/cache/' + scene_id + '.pth')
        pass
    return

def write(filename, model, validator, test_dir, num_scales, augment=True, num_cross_scales=0, scan_scale=50, full_scale=4096):
    """ Predict instance confidence, add additional instances, and write instances """
    scene_id = filename.split('/')[-1].split('_vh_clean')[0]
    coords, colors, edges, semantic_pred, neighbors, num_ori_coords, instance_pred = torch.load(test_dir + '/cache/' + scene_id + '.pth')
    
    coords_inp = torch.from_numpy(np.concatenate([coords, np.zeros((len(coords), 1))], axis=-1)).cuda()
    colors_inp = torch.from_numpy(colors.astype(np.float32) / 127.5 - 1).cuda()
    semantic_pred, neighbor_pred = model(coords_inp, colors_inp)
    semantic_pred = semantic_pred.max(-1)[1].detach().cpu().numpy()

    instance_info = validator.validate(coords, colors[:, :3], instance_pred, semantic_pred)
    
    if True:
        ## Find connected components of certain labels to add additional instances which are likely to be coplanar with larger objects
        semantic_instances, num_semantic_instances = findInstancesSemanticsLabels(edges, semantic_pred, labels=[10, 13, 15, 17, 18])
        if num_semantic_instances > 0:
            instance_info += validator.validate(coords, colors[:, :3], semantic_instances, semantic_pred)
            pass
        pass
    instance_info = [(mask[:num_ori_coords], label, confidence) for mask, label, confidence in instance_info]        

    instance_pred = instance_pred[:num_ori_coords]
    semantic_pred = semantic_pred[:num_ori_coords]
    writeInstances(test_dir, scene_id, instance_pred, semantic_pred, instance_info=instance_info)
    return

if __name__ == '__main__':
    options = parse_args()    
    options.keyname = 'instance'
    #args.keyname += '_' + args.dataset

    if options.suffix != '':
        options.keyname += '_' + options.suffix
        pass
    if options.numScales != 1:
        options.keyname += '_' + str(options.numScales)
        pass    
    
    options.checkpoint_dir = 'checkpoint/' + options.keyname

    filenames = []
    split = options.split

    data_folder = options.dataFolder

    test_dir = 'test/' + options.keyname + '/inference/' + split + '/'
    if not os.path.exists(test_dir):
        os.system("mkdir -p %s"%test_dir)
        pass
    if not os.path.exists(test_dir + '/pred_mask'):
        os.system("mkdir -p %s"%(test_dir + '/pred_mask'))
        pass
    if not os.path.exists(test_dir + '/sem_pred'):
        os.system("mkdir -p %s"%(test_dir + '/sem_pred'))
        pass
    if not os.path.exists(test_dir + '/cache'):
        os.system("mkdir -p %s"%(test_dir + '/cache'))
        pass
    
    label_counts = np.zeros(42)
    with open('datasets/split_' + split + '.txt', 'r') as f:
        for line in f:
            scene_id = line.strip()
            if len(scene_id) < 5 or scene_id[:5] != 'scene':
                continue
            if options.scene_id != '' and options.scene_id not in scene_id:
                continue
            filename = data_folder + '/' + scene_id + '/' + scene_id + '_vh_clean_2.pth'
            info = torch.load(filename)
            
            if os.path.exists(filename) and len(info) == 5:
                filenames.append(filename)
                if split == 'val' and len(filenames) >= options.numTestingImages:
                    break
                pass
            continue
        pass
    
    model = Model(options)
    model.cuda()
    model.eval()

    if options.startEpoch >= 0:
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_' + str(options.startEpoch) + '.pth'))
    else:
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
        pass

    neighbor_model = NeighborGT(options)
    neighbor_model.cuda()

    augmentation_model = CoordAugmentation(options)
    augmentation_model.cuda()
    augmentation_model.eval()

        
    num_scales = options.numScales
    num_cross_scales = options.numCrossScales
    scan_scale = options.scanScale    
    full_scale = options.inputScale    
    
    if options.useCache > 0:
        filenames = [filename for filename in filenames if not os.path.exists(test_dir + '/' + filename + '.txt')]
        pass
    print('num images', len(filenames))    

    assert('predict' in options.task or 'cluster' in options.task or 'write' in options.task)
    
    if 'predict' in options.task:
        for filename in filenames:
            inference(filename, model, neighbor_model, augmentation_model, test_dir, scan_scale=scan_scale, full_scale=full_scale, evaluate_loss=split == 'val', use_normal='normal' in options.suffix)
            continue
        pass

    if 'cluster' in options.task:
        #num_processes = mp.cpu_count()
        num_processes = 8
        p = mp.Pool(processes=num_processes)
        p.map(partial(group, test_dir=test_dir, num_scales=num_scales, num_cross_scales=num_cross_scales, scan_scale=scan_scale), filenames)
        p.close()
        p.join()
        pass

    
    if 'write' in options.task:
        validator = InstanceValidator(options.checkpoint_dir)            
        for filename in filenames:
            write(filename, model, validator, test_dir=test_dir, num_scales=num_scales, num_cross_scales=num_cross_scales, scan_scale=scan_scale)
            continue
        if split == 'val':
            os.system('python scripts/evaluate_semantic_instance.py --pred_path=' + test_dir + ' --num_testing_images=' + str(options.numTestingImages) + ' --scene_id=' + options.scene_id)
            pass
        pass
