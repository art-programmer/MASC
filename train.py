import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import os
import cv2

from utils import *
from options import parse_args

from models.instance import Model, CoordAugmentation, NeighborGT

from datasets.scannet_dataset import ScanNetDataset
from scripts.prepare_data import prepare_data

torch.set_printoptions(threshold=5000)


class_weights = np.zeros(20, dtype=np.float32)
semantic_counts = np.load('datasets/semantic_counts_pixelwise.npy')
for i, x in enumerate(label_subset):
    class_weights[i] = semantic_counts[x]
    continue
class_weights = np.log(class_weights.sum() / class_weights)
class_weights = class_weights / class_weights.sum()
class_weights = torch.from_numpy(class_weights).cuda()
    
    
def main(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    
    model = Model(options)
    model.cuda()
    model.train()

    neighbor_model = NeighborGT(options)
    neighbor_model.cuda()

    augmentation_model = CoordAugmentation(options)
    augmentation_model.cuda()
    augmentation_model.train()

        
    if options.restore == 1:
        print('restore')
        if options.startEpoch >= 0:
            model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_' + str(options.startEpoch) + '.pth'))
        else:
            model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
            pass        
        pass

    dataset_val = ScanNetDataset(options, split='val', random=False)
    if options.task == 'test':
        testOneEpoch(options, model, neighbor_model, augmentation_model, dataset_val, validation=False)
        exit(1)
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr = options.LR)
    
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim.pth'):
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim.pth'))
        pass

    dataset = ScanNetDataset(options, split='train', random=True)
    print('the number of images', len(dataset))    
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=True, num_workers=16)

    for epoch in range(options.numEpochs):
        epoch_losses = []
        data_iterator = tqdm(dataloader, total=int(np.ceil(float(len(dataset)) / options.batchSize)))
        for sample_index, sample in enumerate(data_iterator):
            optimizer.zero_grad()
            
            coords, colors, faces, semantic_gt, instance_gt = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda()

            if 'augment' in options.suffix:
                num_coords = [len(c) for c in coords]
                new_coords = []                
                new_colors = []
                new_instances = []
                instances = instance_gt.unsqueeze(-1)
                for batch_index in range(len(coords)):
                    augmented_coords, augmented_colors, augmented_instances, _ = augmentation_model(coords[batch_index], faces[batch_index], colors[batch_index], instances[batch_index])
                    new_coords.append(torch.cat([coords[batch_index], augmented_coords], dim=0))
                    new_colors.append(torch.cat([colors[batch_index], augmented_colors], dim=0))
                    new_instances.append(torch.cat([instances[batch_index], augmented_instances], dim=0))                                        
                    continue
                coords = torch.stack(new_coords, 0)
                colors = torch.stack(new_colors, 0)
                new_instances = torch.stack(new_instances, 0)
                instance_gt = new_instances[:, :, 0]
                pass
            
            semantic_pred, neighbor_pred = model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])))
            if 'augment' in options.suffix:
                semantic_pred = semantic_pred[:num_coords[0]]
                pass
            
            semantic_loss = torch.nn.functional.cross_entropy(semantic_pred.view((-1, int(semantic_pred.shape[-1]))), semantic_gt.view(-1), weight=class_weights)
            semantic_pred = semantic_pred.max(-1)[1].unsqueeze(0)

            if options.numScales > 0:
                neighbor_gt = neighbor_model(coords.reshape((-1, 4)), instance_gt.reshape((-1, )))
            else:
                neighbor_gt = []
                pass

            if 'mse' not in options.suffix:
                for neighbor in neighbor_pred:
                    neighbor.features = torch.sigmoid(neighbor.features)
                    continue
                pass
            neighbor_losses = []
            for scale in range(len(neighbor_gt)):
                pred = neighbor_pred[scale].features
                gt = neighbor_gt[scale].features[:, :pred.shape[-1]]
                mask = neighbor_gt[scale].features[:, pred.shape[-1]:]
                if 'mse' in options.suffix:
                    neighbor_losses.append(torch.sum(torch.nn.functional.mse_loss(pred, gt, reduce=False) * mask) / torch.clamp(mask.sum(), min=1) / options.numScales)
                else:
                    neighbor_losses.append(torch.sum(torch.nn.functional.binary_cross_entropy(pred, (gt > 0.5).float(), weight=(1 - gt) * (int(options.negativeWeights[scale]) - 1) + 1, reduce=False) * mask) / torch.clamp(mask.sum(), min=1))
                    pass
                continue
            losses = [semantic_loss] + neighbor_losses
            loss = sum(losses)

            loss_values = [l.data.item() for l in losses]
            epoch_losses.append(loss_values)
            status = str(epoch + 1) + ' loss: '
            for l in loss_values:
                status += '%0.5f '%l
                continue
            data_iterator.set_description(status)
            loss.backward()
            optimizer.step()
            
            continue
        print('loss', np.array(epoch_losses).mean(0))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint_' + str(epoch // 10) + '.pth')
            #torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_' + str(epoch // 10) + '.pth')
            pass
        torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint.pth')
        torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim.pth')

        testOneEpoch(options, model, neighbor_model, augmentation_model, dataset_val, validation=True)        
        continue
    return

def testOneEpoch(options, model, neighbor_model, augmentation_model, dataset, validation=True):
    for split in ['pred', 'gt']:
        if not os.path.exists(options.test_dir + '/' + split):
            os.system("mkdir -p %s"%options.test_dir + '/' + split)
            pass
        if not os.path.exists(options.test_dir + '/' + split + '/pred_mask'):
            os.system("mkdir -p %s"%options.test_dir + '/' + split + '/pred_mask')
            pass        
        continue

    model.eval()
    augmentation_model.eval()
    
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=1)
    
    epoch_losses = []    
    data_iterator = tqdm(dataloader, total=int(np.ceil(float(len(dataset)) / options.batchSize)))

    for sample_index, sample in enumerate(data_iterator):
        if sample_index == options.numTestingImages:
            break

        coords, colors, faces, semantic_gt, instance_gt, filenames = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda(), sample[5]

        edges = torch.cat([faces[:, :, [0, 1]], faces[:, :, [1, 2]], faces[:, :, [2, 0]]], dim=1)
        if 'augment' in options.suffix:
            num_coords = [len(c) for c in coords]
            new_coords = []
            new_colors = []
            new_edges = []
            new_instances = []
            instances = instance_gt.unsqueeze(-1)
            for batch_index in range(len(coords)):
                augmented_coords, augmented_colors, augmented_instances, augmented_edges = augmentation_model(coords[batch_index], faces[batch_index], colors[batch_index], instances[batch_index])
                new_coords.append(torch.cat([coords[batch_index], augmented_coords], dim=0))
                new_colors.append(torch.cat([colors[batch_index], augmented_colors], dim=0))
                new_instances.append(torch.cat([instances[batch_index], augmented_instances], dim=0))
                new_edges.append(torch.cat([edges[batch_index], augmented_edges], dim=0))
                continue
            coords = torch.stack(new_coords, 0)
            colors = torch.stack(new_colors, 0)
            new_instances = torch.stack(new_instances, 0)
            instance_gt = new_instances[:, :, 0]
            edges = torch.stack(new_edges, 0)
            pass

        semantic_pred, neighbor_pred = model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])))

        if 'augment' in options.suffix:
            semantic_loss = torch.nn.functional.cross_entropy(semantic_pred[:num_coords[0]].view((-1, int(semantic_pred.shape[-1]))), semantic_gt.view(-1), weight=class_weights)
        else:
            semantic_loss = torch.nn.functional.cross_entropy(semantic_pred.view((-1, int(semantic_pred.shape[-1]))), semantic_gt.view(-1), weight=class_weights)            
            pass
        semantic_pred = semantic_pred.max(-1)[1].unsqueeze(0)
        
        if options.numScales > 0:        
            neighbor_gt = neighbor_model(coords.reshape((-1, 4)), instance_gt.reshape((-1, )))
        else:
            neighbor_gt = []            
            pass

        if 'mse' not in options.suffix:
            for neighbor in neighbor_pred:
                neighbor.features = torch.sigmoid(neighbor.features)
                continue
            pass
        neighbor_losses = []
        for scale in range(len(neighbor_gt)):
            pred = neighbor_pred[scale].features
            gt = neighbor_gt[scale].features[:, :pred.shape[-1]]
            mask = neighbor_gt[scale].features[:, pred.shape[-1]:]
            if 'mse' in options.suffix:
                neighbor_losses.append(torch.sum(torch.nn.functional.mse_loss(pred, gt, reduce=False) * mask) / torch.clamp(mask.sum(), min=1) / options.numScales)
            else:
                neighbor_losses.append(torch.sum(torch.nn.functional.binary_cross_entropy(pred, (gt > 0.5).float(), weight=(1 - gt) * (int(options.negativeWeights[scale]) - 1) + 1, reduce=False) * mask) / torch.clamp(mask.sum(), min=1))
                pass
            continue

        if not validation:
            for c in range(len(neighbor_pred)):
                mask_pred = neighbor_pred[c].features > 0.5
                mask_gt = neighbor_gt[c].features[:, :6] > 0.5
                neighbor_mask = neighbor_gt[c].features[:, 6:] > 0.5
                print(c, (mask_pred * mask_gt * neighbor_mask).sum(), ((1 - mask_pred) * mask_gt * neighbor_mask).sum(), (mask_pred * (1 - mask_gt) * neighbor_mask).sum(), ((1 - mask_pred) * (1 - mask_gt) * neighbor_mask).sum())
                continue
            pass
        
        losses = [semantic_loss] + neighbor_losses
        loss = sum(losses)

        loss_values = [l.data.item() for l in losses]
        epoch_losses.append(loss_values)
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f '%l
            continue
        data_iterator.set_description(status)
        
        #semantic_statistics.append(evaluateSemantics(semantic_pred.detach().cpu().numpy(), semantic_gt.detach().cpu().numpy()))
        if not validation:
            coords = coords.detach().cpu().numpy()[:, :, :3]
            colors = np.clip((colors.detach().cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)        
            semantic_pred = semantic_pred.detach().cpu().numpy()
            semantic_gt = semantic_gt.detach().cpu().numpy()
            semantic_gt[semantic_gt == -100] = -1
            instance_gt = instance_gt.detach().cpu().numpy()
            faces = faces.detach().cpu().numpy()
            edges = edges.detach().cpu().numpy()

            neighbors = neighbor_model.toDense(neighbor_pred)
            neighbors = [neighbor.detach().cpu().numpy() for neighbor in neighbors]
            instance_pred = []
            for batch_index in range(len(filenames)):
                scene_id = filenames[batch_index].split('/')[-1].split('_vh_clean')[0]
                instances, intermediate_instances = findInstances(coords[batch_index], edges[batch_index], np.zeros(len(coords[batch_index])).astype(np.int32), neighbors, options.numScales, options.numCrossScales, cache_filename=options.test_dir + '/pred/' + scene_id + '.txt' if options.useCache else '', scene_id=scene_id)
                instance_pred.append(instances)
                continue

            instance_info_array = []
            if options.useCache <= 1:
                for batch_index in range(len(coords)):
                    #writeSemantics(options.test_dir + '/sem_pred/' + str(sample_index * options.batchSize + batch_index) + '.txt', semantic_pred[batch_index])
                    #writeSemantics(options.test_dir + '/sem_gt/' + str(sample_index * options.batchSize + batch_index) + '.txt', semantic_gt[batch_index])
                    scene_id = filenames[batch_index].split('/')[-1].split('_vh_clean')[0]

                    instances = instance_pred[batch_index]
                    semantics = semantic_pred[batch_index]
                    num_ori_coords = num_coords[batch_index]                    
                    if 'augment' in options.suffix:
                        instance_labels, counts = np.unique(instances[:num_ori_coords], return_counts=True)
                        valid_labels = instance_labels[counts > 100]
                        valid_labels = valid_labels[valid_labels >= 0]
                        #print('num valid instances', len(valid_labels))
                        label_map = np.full(instances.max() + 1, fill_value=-1, dtype=np.int32)
                        for index, label in enumerate(valid_labels):
                            label_map[label] = index
                            continue
                        instances = label_map[instances]
                        pass                        

                    instance_info = []
                    semantic_instances, num_semantic_instances = findInstancesSemanticsLabels(edges[batch_index], semantics)
                    if num_semantic_instances > 0:
                        instances[semantic_instances >= 0] = semantic_instances[semantic_instances >= 0] + instances.max() + 1
                        pass
                    
                    instances = instances[:num_ori_coords]
                    semantics = semantics[:num_ori_coords]

                    print('num instances', len(np.unique(instance_gt[batch_index])), instances.max() + 1)
                    #writeInstances(options.test_dir + '/gt/', scene_id, instance_pred[batch_index], semantic_gt[batch_index])
                    
                    instance_info = writeInstances(options.test_dir + '/pred/', scene_id, instances, semantics, instance_info)
                    instance_labels = np.zeros(num_ori_coords, dtype=np.int32)
                    for mask, label, confidence in instance_info:
                        print(label, confidence)
                        instance_labels[mask] = label
                        continue

                    unique_instances, first_indices, new_instance_gt = np.unique(instance_gt[batch_index], return_index=True, return_inverse=True)
                    instance_semantics_gt = mapper[semantic_gt[batch_index][first_indices]]
                    #print('num', (instance_semantics_gt == 8).sum())
                    instance_labels_gt = instance_semantics_gt[new_instance_gt]
                    visualizeExample(options, coords[batch_index], faces[batch_index], colors[batch_index], num_ori_coords, [('pred', {'semantic': semantics, 'instance': instances, 'instance_label': instance_labels}), ('gt', {'semantic': semantic_gt[batch_index], 'instance': new_instance_gt, 'instance_label': instance_labels_gt})], index_offset=sample_index)
                    continue
                pass
            if options.visualizeMode == 'debug':
                exit(1)
                pass
            pass
        continue
    print('validation loss', np.array(epoch_losses).mean(0))
    #semantic_statistics = np.array(semantic_statistics).sum(0)
    #print(semantic_statistics[:len(semantic_statistics) // 2].astype(np.float32) / np.maximum(semantic_statistics[len(semantic_statistics) // 2:], 1))
    model.train()
    augmentation_model.train()    
    return


def visualizeExample(options, coords, faces, colors, num_coords, dicts, index_offset=0, prefix=''):
    """ Visualize results for one example """
    write_ply_color(options.test_dir + '/' + str(index_offset) + '_input_color.ply', coords, faces, colors[:, :3])
    write_ply_color(options.test_dir + '/' + str(index_offset) + '_input_normal.ply', coords, faces, colors[:, 3:6])        
    for name, result_dict in dicts:
        semantics = result_dict['semantic']

        filename = options.test_dir + '/' + str(index_offset) + '_' + name + '_semantic.ply'
        write_ply_label(filename, coords[:len(semantics)], faces, semantics)

        if 'instance' in result_dict:
            instances = result_dict['instance']

            filename = options.test_dir + '/' + str(index_offset) + '_' + name + '_instance.ply'
            #print(name, len(instances), np.unique(instances, return_counts=True))
            write_ply_label(filename, coords[:len(instances)], faces, instances, debug_index=-1)

            if False:
                filename = options.test_dir + '/' + str(index_offset) + '_' + name + '_edge.ply'
                write_ply_edge(filename, coords, faces, instances)
                pass
            pass
        print(result_dict.keys())
        if 'instance_label' in result_dict:
            filename = options.test_dir + '/' + str(index_offset) + '_' + name + '_instance_semantic.ply'
            write_ply_label(filename, coords[:num_coords], faces, result_dict['instance_label'][:num_coords], debug_index=-1)
            pass                
        continue
    return

if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'instance'

    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    if args.numScales != 1:
        args.keyname += '_' + str(args.numScales)
        pass    
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = 'test/' + args.keyname

    print('keyname=%s task=%s started'%(args.keyname, args.task))

    ## Prepare ScanNet data
    if args.task == 'prepare':        
        prepare_data(args)
        exit(1)
        pass
    main(args)
