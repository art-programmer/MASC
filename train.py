import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import os
import cv2

from utils import *
from options import parse_args

from models.instance import Model, InstanceGTScale, CoordAugmentation, NeighborGT, SemanticClassifier

from datasets.scannet_dataset import ScanNetDataset
from train_semantics import InstanceClassifier

torch.set_printoptions(threshold=5000)

CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
ID_TO_LABEL = {}
ID_TO_LABEL[0] = 'wall'
ID_TO_LABEL[1] = 'floor'
ID_TO_LABEL[20] = 'invalid'
ID_TO_LABEL[-1] = 'invalid'    

DEBUG_INFO = ['cabinet', 2, 3, 'bed', 3, 4, 'chair', 4, 5, 'sofa', 5, 6, 'table', 6, 7, 'door', 7, 8, 'window', 8, 9, 'bookshelf', 9, 10, 'picture', 10, 11, 'counter', 11, 12, 'desk', 12, 14, 'curtain', 13, 16, 'refrigerator', 14, 24, 'shower curtain', 15, 28, 'toilet', 16, 33, 'sink', 17, 34, 'bathtub', 18, 36, 'otherfurniture', 19, 39]

for i in range(len(VALID_CLASS_IDS)):
    #ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
    ID_TO_LABEL[i + 2] = CLASS_LABELS[i]
    continue



def main(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    
    if 'maxpool' not in options.suffix:
        class_weights = np.zeros(20, dtype=np.float32)
        semantic_counts = np.load('datasets/semantic_counts_pixelwise.npy')
        for i, x in enumerate(label_subset):
            class_weights[i] = semantic_counts[x]
            continue
        #class_weights /= class_weights.sum()
        #class_weights = 1 / np.log(1.2 + class_weights)

        class_weights = np.log(class_weights.sum() / class_weights)
        #class_weights = class_weights.sum() / class_weights
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.from_numpy(class_weights).cuda()
        #CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    else:
        class_weights = np.zeros(21, dtype=np.float32)
        semantic_counts = np.load('datasets/semantic_counts.npy')
        for i, x in enumerate(label_subset):
            class_weights[i] = semantic_counts[x]
            continue
        class_weights[20] = 1201 #num of images
        class_weights = np.log(class_weights.sum() / class_weights)
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.from_numpy(class_weights).cuda()
        pass
    
    
    model = Model(options)
    model.cuda()
    model.train()

    #neighbor_model = InstanceGT(options.inputScale, 6)
    if 'soft' in options.suffix:
        neighbor_model = InstanceGTScale(options)
    else:
        neighbor_model = NeighborGT(options)
        pass
    neighbor_model.cuda()

    augmentation_model = CoordAugmentation(options)
    augmentation_model.cuda()
    augmentation_model.train()

    if 'maxpool' in options.suffix:
        semantic_model = SemanticClassifier()
        semantic_model.cuda()
        semantic_model.train()
    else:
        semantic_model = None
        pass
        
    if options.restore == 1:
        print('restore')
        if options.startEpoch >= 0:
            model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_' + str(options.startEpoch) + '.pth'))
            if 'maxpool' in options.suffix:
                semantic_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_semantic_' + str(options.startEpoch) + '.pth'))
                pass
        else:
            model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
            if 'maxpool' in options.suffix:
                semantic_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_semantic.pth'))
                pass
            pass        
    elif options.restore == 2:
        state_dict = torch.load(options.checkpoint_dir + '/checkpoint.pth')
        state = model.state_dict()
        new_state_dict = {k: v for k, v in state_dict.items() if k in state and v.shape == state[k].shape}
        state.update(new_state_dict)
        model.load_state_dict(state)
    elif options.restore == 3:
        state_dict = torch.load('checkpoint/instance_normal_augment_2/checkpoint.pth')
        state = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in state and v.shape == state[k].shape:
                new_state_dict[k] = v
            elif state[k].shape[0] < v.shape[0]:
                new_state_dict[k] = v[:state[k].shape[0]]
            else:
                new_state_dict[k] = torch.cat([v, v[:1].repeat((state[k].shape[0] - v.shape[0], 1))], dim=0)
                pass
            continue
        state.update(new_state_dict)
        model.load_state_dict(state)
    elif options.restore == 4:
        state_dict = torch.load('../ScanNet/unet_scale20_m16_rep1_notResidualBlocks-000000413-unet.pth')
        state = model.state_dict()
        new_state_dict = {k: v for k, v in state_dict.items() if k in state and v.shape == state[k].shape}
        state.update(new_state_dict)
        model.load_state_dict(state)
    elif options.restore == 5:
        if options.startEpoch >= 0:
            model.load_state_dict(torch.load(options.checkpoint_dir.replace('_augment', '') + '/checkpoint_' + str(options.startEpoch) + '.pth'))
        else:
            model.load_state_dict(torch.load(options.checkpoint_dir.replace('_augment', '') + '/checkpoint.pth'))
            pass                
        pass

    dataset_test = ScanNetDataset(options, split='val', random=False)
    print(len(dataset_test))
    if options.task == 'test':
        testOneEpoch(options, model, neighbor_model, augmentation_model, semantic_model, dataset_test, validation=False)
        exit(1)
        pass

    if 'maxpool' in options.suffix:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(semantic_model.parameters()), lr = options.LR)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = options.LR)
        pass
    
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
            
            coords, colors, faces, semantic_gt, instance_gt, invalid_instance_gt = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda(), sample[5].cuda()

            if 'augment' in options.suffix:
                num_coords = [len(c) for c in coords]
                new_coords = []                
                new_colors = []
                new_instances = []
                instances = torch.stack([instance_gt, invalid_instance_gt], dim=-1)
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
                invalid_instance_gt = new_instances[:, :, 1]                
                pass
            
            semantic_pred, neighbor_pred = model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])))
            #semantic_pred = semantic_pred.reshape((len(coords), -1))
            if 'maxpool' in options.suffix:
                semantic_pred, semantic_gt, instance_masks = semantic_model(semantic_pred, instance_gt[0], invalid_instance_gt[0], semantic_gt[0])
                if False:
                    coords = coords.detach().cpu().numpy()
                    faces = faces.detach().cpu().numpy()                                            
                    for index, instance_mask in enumerate(instance_masks):
                        print(index, semantic_gt[index])
                        instance_mask = instance_mask.int().detach().cpu().numpy()
                        print(coords[0].shape, faces[0].max(), instance_mask.shape)
                        write_ply_label('test/instance_' + str(index) + '.ply', coords[0][:, :3], faces[0], instance_mask)
                        continue
                    exit(1)
                    pass
            elif 'augment' in options.suffix:
                semantic_pred = semantic_pred[:num_coords[0]]
                pass
            
            semantic_loss = torch.nn.functional.cross_entropy(semantic_pred.view((-1, int(semantic_pred.shape[-1]))), semantic_gt.view(-1), weight=class_weights)
            semantic_pred = semantic_pred.max(-1)[1].unsqueeze(0)

            #print(neighbor_pred[scale].features.shape, neighbor_gt[scale].shape)

            #neighbor_loss = sum([torch.nn.functional.mse_loss(neighbor_pred[scale].features, torch.cat([neighbor_gt[batch_index][scale][0] for batch_index in range(len(neighbor_gt))], dim=0).cuda()) for scale in range(len(neighbor_gt))])

            if options.numScales > 0:
                neighbor_gt = neighbor_model(coords.reshape((-1, 4)), instance_gt.reshape((-1, )))
            else:
                neighbor_gt = []
                pass
            #print(torch.nn.functional.mse_loss(neighbor_pred[0].features, (1 - neighbor_gt[0].features) * 10, reduce=False).shape, neighbor_mask.shape)
            #exit(1)

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

            if sample_index % 500 == 0 and False:
                #visualizeBatch(options, images.detach().cpu().numpy(), [('gt', {'corner': corner_gt.detach().cpu().numpy(), 'icon': icon_gt.detach().cpu().numpy(), 'room': room_gt.detach().cpu().numpy()}), ('pred', {'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 'room': room_pred.max(-1)[1].detach().cpu().numpy()})])
                coords = coords.detach().cpu().numpy()[:, :, :3]
                print(colors.shape)
                colors = np.clip((colors.detach().cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
                print(coords.min(), coords.max())
                print(semantic_gt.min(), semantic_gt.max())
                semantic_gt = semantic_gt.detach().cpu().numpy()
                semantic_pred = semantic_pred.detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()

                write_ply_color('test/input_normal.ply', coords[0], faces[0], colors[0][:, 3:6])
                #write_ply_label('test/input_semantic.ply', coords[0], faces[0], semantic_gt[0])
                exit(1)
                visualizeBatch(options, coords, faces, colors, [('gt', {'semantic': semantic_gt}), ('pred', {'semantic': semantic_pred})])
                if options.visualizeMode == 'debug':
                    exit(1)
                    pass
            continue
        print('loss', np.array(epoch_losses).mean(0))
        if True:
            if epoch % 10 == 0:
                torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint_' + str(epoch // 10) + '.pth')
                if 'maxpool' in options.suffix:
                    torch.save(semantic_model.state_dict(), options.checkpoint_dir + '/checkpoint_semantic_' + str(epoch // 10) + '.pth')
                    pass
                #torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_' + str(epoch // 10) + '.pth')
                pass
            torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint.pth')
            if 'maxpool' in options.suffix:
                torch.save(semantic_model.state_dict(), options.checkpoint_dir + '/checkpoint_semantic.pth')
                pass
            torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim.pth')
            pass
        testOneEpoch(options, model, neighbor_model, augmentation_model, semantic_model, dataset_test, validation=True)        
        #testOneEpoch(options, model, dataset_test)        
        continue
    return

def testOneEpoch(options, model, neighbor_model, augmentation_model, semantic_model, dataset, validation=True):
    for split in ['pred', 'gt']:
        if not os.path.exists(options.test_dir + '/' + split):
            os.system("mkdir -p %s"%options.test_dir + '/' + split)
            pass
        if not os.path.exists(options.test_dir + '/' + split + '/pred_mask'):
            os.system("mkdir -p %s"%options.test_dir + '/' + split + '/pred_mask')
            pass        
        continue

    #print(model)
    #model.eval()
    augmentation_model.eval()
    
    # bn = list(list(model.children())[0].children())[3]
    # print(bn.running_var, bn.running_mean)
    # exit(1)
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=1)
    
    epoch_losses = []    
    data_iterator = tqdm(dataloader, total=int(np.ceil(float(len(dataset)) / options.batchSize)))

    classifier = InstanceClassifier()
    #semantic_statistics = []
    #instance_statistics = []    
    for sample_index, sample in enumerate(data_iterator):
        if sample_index == options.numTestingImages:
            break

        coords, colors, faces, semantic_gt, instance_gt, invalid_instance_gt, filenames = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda(), sample[5].cuda(), sample[6]

        edges = torch.cat([faces[:, :, [0, 1]], faces[:, :, [1, 2]], faces[:, :, [2, 0]]], dim=1)
        if 'augment' in options.suffix:
            num_coords = [len(c) for c in coords]
            new_coords = []
            new_colors = []
            new_edges = []
            new_instances = []
            instances = torch.stack([instance_gt, invalid_instance_gt], dim=-1)            
            for batch_index in range(len(coords)):
                augmented_coords, augmented_colors, augmented_instances, augmented_edges = augmentation_model(coords[batch_index], faces[batch_index], colors[batch_index], instances[batch_index])
                new_coords.append(torch.cat([coords[batch_index], augmented_coords], dim=0))
                new_colors.append(torch.cat([colors[batch_index], augmented_colors], dim=0))
                new_instances.append(torch.cat([instances[batch_index], augmented_instances], dim=0))
                #new_instances.append(torch.cat([instance_gt[batch_index], -1 * torch.ones(augmented_instances.shape).cuda().long()], dim=0))
                new_edges.append(torch.cat([edges[batch_index], augmented_edges], dim=0))
                # new_coords.append(augmented_coords)
                # new_colors.append(augmented_colors)
                # new_instances.append(augmented_instances)
                continue
            coords = torch.stack(new_coords, 0)
            colors = torch.stack(new_colors, 0)
            new_instances = torch.stack(new_instances, 0)
            instance_gt = new_instances[:, :, 0]
            invalid_instance_gt = new_instances[:, :, 1]                
            edges = torch.stack(new_edges, 0)
            pass

        semantic_pred, neighbor_pred = model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])))

        #semantic_pred = semantic_pred.reshape((len(coords), -1))
        if 'maxpool' in options.suffix:
            semantic_pred, instance_semantic_gt, _ = semantic_model(semantic_pred, instance_gt[0], invalid_instance_gt[0], semantic_gt[0])
            semantic_loss = torch.nn.functional.cross_entropy(semantic_pred.view((-1, int(semantic_pred.shape[-1]))), instance_semantic_gt.view(-1))
            pass
        elif 'augment' in options.suffix:
            semantic_loss = torch.nn.functional.cross_entropy(semantic_pred[:num_coords[0]].view((-1, int(semantic_pred.shape[-1]))), semantic_gt.view(-1))
        else:
            semantic_loss = torch.nn.functional.cross_entropy(semantic_pred.view((-1, int(semantic_pred.shape[-1]))), semantic_gt.view(-1))            
            pass
        semantic_pred = semantic_pred.max(-1)[1].unsqueeze(0)
        
        if options.numScales > 0:        
            neighbor_gt = neighbor_model(coords.reshape((-1, 4)), instance_gt.reshape((-1, )))
        else:
            neighbor_gt = []            
            pass

        if False:
            for scale in range(len(neighbor_gt)):
                neighbors = neighbor_gt[scale]
                neighbor_coords = neighbors.get_spatial_locations().detach().cpu().numpy()[:, :3]
                neighbor_labels = neighbors.features[:, :6].detach().cpu().numpy()
                neighbor_masks = neighbors.features[:, 6:].detach().cpu().numpy()
                write_ply_neighbor('test/input_neighbor_' + str(scale) + '.ply', neighbor_coords, neighbor_labels, neighbor_masks, size=options.inputScale // pow(2, scale))
                continue

            coords = coords.detach().cpu().numpy()[:, :, :3]
            colors = np.clip((colors.detach().cpu().numpy() + 1) * 127.5, 0, 255).astype(np.uint8)
            instance_gt = instance_gt.detach().cpu().numpy()
            faces = faces.detach().cpu().numpy()
            edges = edges.detach().cpu().numpy()            
            #augmented_edges = augmented_edges.detach().cpu().numpy()
            #faces = np.array([[[0, 1, 2]]])
            write_ply_color('test/input_normal.ply', coords[0], faces[0], colors[0][:, 3:6])
            write_ply_edge('test/input_edge.ply', coords[0], edges[0], instance_gt[0])
            exit(1)
        
            
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

        # print(neighbor_losses)
        # print(semantic_pred.shape)
        # print((torch.abs(semantic_pred.view(-1)[:len(semantic_gt.view(-1))] - semantic_gt.view(-1)) <= 1).sum())
        # print(semantic_loss)
        # exit(1)
        
        # print(coords[0].min(0)[0], coords[0].max(0)[0], colors.min(), colors.max())
        # print(np.unique(semantic_pred.detach().cpu().numpy()), (neighbor_pred[0].features < 0.5).sum(), neighbor_pred[0].features.shape, neighbor_pred[0].features.min(), neighbor_pred[0].features.max())
        # exit(1)
                
        if not validation:
            for c in range(len(neighbor_pred)):
                mask_pred = neighbor_pred[c].features > 0.5
                mask_gt = neighbor_gt[c].features[:, :6] > 0.5
                neighbor_mask = neighbor_gt[c].features[:, 6:] > 0.5
                print(c, (mask_pred * mask_gt * neighbor_mask).sum(), ((1 - mask_pred) * mask_gt * neighbor_mask).sum(), (mask_pred * (1 - mask_gt) * neighbor_mask).sum(), ((1 - mask_pred) * (1 - mask_gt) * neighbor_mask).sum())
                #print(torch.cat([neighbor_pred[0].features, neighbor_gt[0].features], dim=-1)[:10])
                #exit(1)
                continue
            pass
        #neighbor_loss = sum([torch.nn.functional.mse_loss(neighbor_pred[scale].features, torch.cat([neighbor_gt[batch_index][scale][0] for batch_index in range(len(neighbor_gt))], dim=0).cuda()) for scale in range(len(neighbor_gt))])
        
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
            #print([neighbor.shape for neighbor in neighbor_gt])

            #neighbors = [neighbor_model.toDense(neighbor).detach().cpu().numpy() for neighbor in neighbor_gt]
            neighbors = neighbor_model.toDense(neighbor_pred)
            neighbors = [neighbor.detach().cpu().numpy() for neighbor in neighbors]
            instance_pred = []
            for batch_index in range(len(filenames)):
                scene_id = filenames[batch_index].split('/')[-1].split('_vh_clean')[0]
                instances, intermediate_instances = findInstances(coords[batch_index], edges[batch_index], np.zeros(len(coords[batch_index])).astype(np.int32), neighbors, options.numScales, options.numCrossScales, cache_filename=options.test_dir + '/pred/' + scene_id + '.txt' if options.useCache else '', scene_id=scene_id)
                instance_pred.append(instances)
                if False:
                    intermediate_instances = [instances for instances in intermediate_instances]
                    for index, instances in enumerate(intermediate_instances):
                        filename = options.test_dir + '/intermediate_' + str(index) + '.ply'
                        write_ply_label(filename, coords[0], faces[0], instances)
                        continue
                    pass
                #instance_pred.append(findInstancesSemantics(options, faces[batch_index], semantic_pred[batch_index]))
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

                    if False:
                        instance_inp = torch.from_numpy(instances).cuda() + 1
                        semantic_pred, _, instance_masks = semantic_model(semantic_pred, instance_inp)
                        semantic_pred = torch.nn.functional.softmax(semantic_pred, dim=-1)
                        confidences, labels = semantic_pred.max(-1)

                        #print(semantic_gt.shape, labels.shape, confidences.shape, semantics.shape, semantics[torch.arange(len(semantic_gt)).cuda().long(), semantic_gt].shape)
                        #counts = torch.Tensor([len(instance_mask) for instance_mask in instance_masks]).long().cuda()
                        confidences, labels, semantic_pred = confidences.detach().cpu().numpy(), labels.detach().cpu().numpy(), semantic_pred.detach().cpu().numpy()
                        instance_masks = [instance_mask.detach().cpu().numpy() for instance_mask in instance_masks]

                        instance_semantic_gt = []
                        for instance_mask in instance_masks:
                            print(instance_mask.shape)
                            info = np.unique(semantic_gt[batch_index][instance_mask[:num_ori_coords] > 0.5], return_counts=True)
                            instance_semantic_gt.append(info[0][info[1].argmax()])
                            continue
                        instance_semantic_gt = np.array(instance_semantic_gt)
                        instance_semantic_gt[instance_semantic_gt < 0] = 20
                        counts = np.stack([instance_mask.sum() for instance_mask in instance_masks])
                        info = np.stack([np.arange(len(instance_masks)), mapper[instance_semantic_gt], mapper[labels], np.round(confidences * 10), np.round(semantic_pred[np.arange(len(instance_semantic_gt)), instance_semantic_gt] * 10), counts], axis=-1).astype(np.int32)
                        print(info)
                        print(info[np.logical_and(info[:, -1] > 1000, info[:, 1] != info[:, 2])])
                        labels = instance_semantic_gt

                        instance_masks = [instance_mask[:num_ori_coords] for instance_mask in instance_masks]
                        instance_info = [(instance_mask, label, confidence) for instance_mask, label, confidence in zip(instance_masks, labels, confidences)]
                        semantics = labels[np.maximum(instances - 1, 0)]
                        semantics[instances < 0] = -1
                    elif False:
                        print('classify', len(coords[batch_index]), len(instances), num_ori_coords)
                        instance_info = classifier.classify(coords[batch_index][:num_ori_coords], colors[batch_index][:, :3][:num_ori_coords], instances[:num_ori_coords], semantics[:num_ori_coords])
                        instance_info = [(mask[:num_ori_coords], label, confidence) for mask, label, confidence in instance_info]
                    else:
                        instance_info = []
                        pass

                    semantic_instances, num_semantic_instances = findInstancesSemanticsLabels(edges[batch_index], semantics)
                    if num_semantic_instances > 0:
                        instances[semantic_instances >= 0] = semantic_instances[semantic_instances >= 0] + instances.max() + 1
                        pass
                    #instances = semantic_instances
                    
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
                    visualizeExample(options, coords[batch_index], faces[batch_index], colors[batch_index], num_ori_coords, [('pred', {'semantic': semantics, 'instance': instances, 'instance_label': instance_labels}), ('gt', {'semantic': semantic_gt[batch_index], 'instance': new_instance_gt, 'instance_label': instance_labels_gt})])
                    continue
                pass
            #visualizeBatch(options, coords, faces, colors, num_coords, [('gt', {'semantic': semantic_gt, 'instance': instance_gt}), ('pred', {'semantic': semantic_pred, 'instance': instance_pred})], instance_info_array)
            
            #exit(1)
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

# def evaluateSemantics(semantic_pred, semantic_gt):
#     correct_mask = semantic_pred == semantic_gt
#     valid_mask = semantic_gt >= 0
#     #accuracy = float(correct_mask.sum()) / valid_mask.sum()
#     correct_predictions = semantic_gt[correct_mask]
#     valid_targets = semantic_gt[valid_mask]
#     num_correct_predictions = (np.expand_dims(correct_predictions, -1) == np.arange(20)).sum(0)
#     num_targets = (np.expand_dims(valid_targets, -1) == np.arange(20)).sum(0)
#     statistics = [num_correct_predictions.sum()] + num_correct_predictions.tolist() + [num_targets.sum()] + num_targets.tolist()
#     return statistics

# def visualizeBatch(options, coords, faces, colors, num_coords, dicts, instance_info_array, indexOffset=0, prefix=''):
#     #cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
#     #pointColorMap = ColorPalette(20).getColorMap()
#     #images = ((images.transpose((0, 2, 3, 1)) + 0.5) * 255).astype(np.uint8)
#     for batch_index in range(len(coords)):
#         #cv2.imwrite(filename, image)
#         write_ply_color(options.test_dir + '/' + str(indexOffset + batch_index) + '_input_color.ply', coords[batch_index], faces[batch_index], colors[batch_index][:, :3])
#         write_ply_color(options.test_dir + '/' + str(indexOffset + batch_index) + '_input_normal.ply', coords[batch_index], faces[batch_index], colors[batch_index][:, 3:6])        
#         for name, result_dict in dicts:
#             semantics = result_dict['semantic'][batch_index]
            
#             filename = options.test_dir + '/' + str(indexOffset + batch_index) + '_' + name + '_semantic.ply'
#             write_ply_label(filename, coords[batch_index][:len(semantics)], faces[batch_index], semantics)

#             if 'instance' in result_dict:
#                 instances = result_dict['instance'][batch_index]

#                 filename = options.test_dir + '/' + str(indexOffset + batch_index) + '_' + name + '_instance.ply'
#                 print(name, len(instances), np.unique(instances, return_counts=True))
#                 write_ply_label(filename, coords[batch_index][:len(instances)], faces[batch_index], instances, debug_index=3)

#                 if False:
#                     filename = options.test_dir + '/' + str(indexOffset + batch_index) + '_' + name + '_edge.ply'                    
#                     write_ply_edge(filename, coords[batch_index], faces[batch_index], instances)
#                     pass
                
#                 if name == 'pred':
#                     instance_info = instance_info_array[batch_index]
#                     if len(instance_info) == 0:
#                         instance_semantics = {}
#                         for instance, semantic in zip(instances, semantics):
#                             if instance not in instance_semantics:
#                                 instance_semantics[instance] = []
#                                 pass
#                             instance_semantics[instance].append(semantic)
#                             continue                                    
#                         instance_semantic_map = np.zeros(max(list(instance_semantics.keys())) + 1, dtype=np.int32)
#                         for instance, semantic_labels in instance_semantics.items():
#                             semantic_labels, counts = np.unique(semantic_labels, return_counts=True)
#                             instance_semantic_map[instance] = semantic_labels[counts.argmax()]
#                             continue
#                         print(instance_semantic_map)
                        
#                         instance_labels = instance_semantic_map[instances]
#                         instance_labels = instance_labels[:num_coords[batch_index]]

#                         if False:
#                             labels = np.unique(instances[instance_labels == 3])
#                             print(labels)
#                             instance_gt = dicts[0][1]['instance'][batch_index]
#                             instance_gt = instance_gt[:num_coords[batch_index]]                    
#                             for label in labels:
#                                 labels_gt = instance_gt[instances == label]
#                                 labels_gt, counts = np.unique(labels_gt, return_counts=True)
#                                 print(np.stack([labels_gt, counts], axis=-1))
#                                 continue
#                             pass
#                     else:
#                         instance_labels = np.zeros(num_coords[batch_index], dtype=np.int32)
#                         for mask, label, confidence in instance_info:
#                             instance_labels[mask] = label
#                             continue
#                         pass
#                     #     filename = options.test_dir + '/' + str(indexOffset + batch_index) + '_' + name + '_instance_semantic_' + str(label) + '.ply'
#                     #     write_ply_label(filename, coords[batch_index][:num_coords[batch_index]], faces[batch_index], instances[:num_coords[batch_index]], debug_index=label)
#                     #     continue
                    
#                     filename = options.test_dir + '/' + str(indexOffset + batch_index) + '_' + name + '_instance_semantic.ply'
#                     write_ply_label(filename, coords[batch_index][:num_coords[batch_index]], faces[batch_index], instance_labels[:num_coords[batch_index]])
#                     pass                
#                 pass
#             #cv2.imwrite(filename.replace('image', info + '_' + name), drawSegmentationImage(result_dict[info][batch_index], blackIndex=0, blackThreshold=0.5))
#             continue
#         continue
#     return


def visualizeExample(options, coords, faces, colors, num_coords, dicts, indexOffset=0, prefix=''):
    #cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
    #pointColorMap = ColorPalette(20).getColorMap()
    #images = ((images.transpose((0, 2, 3, 1)) + 0.5) * 255).astype(np.uint8)
    write_ply_color(options.test_dir + '/' + str(indexOffset) + '_input_color.ply', coords, faces, colors[:, :3])
    write_ply_color(options.test_dir + '/' + str(indexOffset) + '_input_normal.ply', coords, faces, colors[:, 3:6])        
    for name, result_dict in dicts:
        semantics = result_dict['semantic']

        filename = options.test_dir + '/' + str(indexOffset) + '_' + name + '_semantic.ply'
        write_ply_label(filename, coords[:len(semantics)], faces, semantics)

        if 'instance' in result_dict:
            instances = result_dict['instance']

            filename = options.test_dir + '/' + str(indexOffset) + '_' + name + '_instance.ply'
            #print(name, len(instances), np.unique(instances, return_counts=True))
            write_ply_label(filename, coords[:len(instances)], faces, instances, debug_index=-1)

            if False:
                filename = options.test_dir + '/' + str(indexOffset) + '_' + name + '_edge.ply'
                write_ply_edge(filename, coords, faces, instances)
                pass
            pass
        print(result_dict.keys())
        if 'instance_label' in result_dict:
            filename = options.test_dir + '/' + str(indexOffset) + '_' + name + '_instance_semantic.ply'
            write_ply_label(filename, coords[:num_coords], faces, result_dict['instance_label'][:num_coords], debug_index=-1)
            pass                
        continue
    return

if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'instance'
    #args.keyname += '_' + args.dataset

    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    if args.numScales != 1:
        args.keyname += '_' + str(args.numScales)
        pass    
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = 'test/' + args.keyname

    print('keyname=%s task=%s started'%(args.keyname, args.task))

    main(args)
