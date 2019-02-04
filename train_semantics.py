import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import os
import cv2

from utils import *
from options import parse_args

from models.instance import Model, InstanceGTScale, CoordAugmentation, NeighborGT, Classifier

from datasets.scannet_dataset import ScanNetDataset

torch.set_printoptions(threshold=5000)

CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
ID_TO_LABEL = {}
ID_TO_LABEL[0] = 'wall'
ID_TO_LABEL[1] = 'floor'
ID_TO_LABEL[20] = 'invalid'
ID_TO_LABEL[-1] = 'invalid'

for i in range(len(VALID_CLASS_IDS)):
    #ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
    ID_TO_LABEL[i + 2] = CLASS_LABELS[i]
    continue


class_weights = np.zeros(21, dtype=np.float32)
semantic_counts = np.load('datasets/semantic_counts_pixelwise.npy')
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    class_weights[i] = semantic_counts[x]
    continue
class_weights[20] = semantic_counts.sum() - class_weights.sum()
#class_weights /= class_weights.sum()
#class_weights = 1 / np.log(1.2 + class_weights)

class_weights = np.log(class_weights.sum() / class_weights)
#class_weights = class_weights.sum() / class_weights
class_weights = class_weights / class_weights.sum()
class_weights = torch.from_numpy(class_weights).cuda()
print(class_weights)
#CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']

    
def main(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass


    # model = Model(options)
    # model.cuda()
    # model.train()

    # #neighbor_model = InstanceGT(options.inputScale, 6)
    # if 'soft' in options.suffix:
    #     neighbor_model = InstanceGTScale(options)
    # else:
    #     neighbor_model = NeighborGT(options)
    #     pass
    # neighbor_model.cuda()

    # augmentation_model = CoordAugmentation(options)
    # augmentation_model.cuda()

    semantic_model = Classifier(options.inputScale, 'normal' in options.suffix)
    semantic_model.cuda()    
    
    if options.restore == 1:
        print('restore')
        if options.startEpoch >= 0:
            semantic_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_semantic_' + str(options.startEpoch) + '.pth'))
        else:
            semantic_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_semantic.pth'))
            pass        
        pass
    
    
    dataset_test = ScanNetDataset(options, split='val', random=False)
    print(len(dataset_test))
    if options.task == 'test':
        testOneEpoch(options, semantic_model, dataset_test, validation=False)
        exit(1)
    
    optimizer = torch.optim.Adam(semantic_model.parameters(), lr = options.LR)
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim_semantic.pth'):
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim_semantic.pth'))
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

            semantic_pred = semantic_model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])))
            #semantic_pred = semantic_pred.reshape((len(coords), -1))
            semantic_loss = torch.nn.functional.cross_entropy(semantic_pred.view((-1, int(semantic_pred.shape[-1]))), semantic_gt.view(-1), weight=class_weights)
            semantic_pred = semantic_pred.max(-1)[1].unsqueeze(0)

            losses = [semantic_loss]
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
        if True:
            if epoch % 10 == 0:
                torch.save(semantic_model.state_dict(), options.checkpoint_dir + '/checkpoint_semantic_' + str(epoch // 10) + '.pth')
                torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_semantic_' + str(epoch // 10) + '.pth')
                pass
            torch.save(semantic_model.state_dict(), options.checkpoint_dir + '/checkpoint_semantic.pth')
            torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_semantic.pth')
            pass
        testOneEpoch(options, semantic_model, dataset_test, validation=True)        
        #testOneEpoch(options, model, dataset_test)        
        continue
    return

def testOneEpoch(options, semantic_model, dataset, validation=True):
    for split in ['pred', 'gt']:
        if not os.path.exists(options.test_dir + '/' + split):
            os.system("mkdir -p %s"%options.test_dir + '/' + split)
            pass
        if not os.path.exists(options.test_dir + '/' + split + '/pred_mask'):
            os.system("mkdir -p %s"%options.test_dir + '/' + split + '/pred_mask')
            pass        
        continue

    #print(model)
    semantic_model.eval()
    
    # bn = list(list(model.children())[0].children())[3]
    # print(bn.running_var, bn.running_mean)
    # exit(1)
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=1)
    
    epoch_losses = []    
    data_iterator = tqdm(dataloader, total=int(np.ceil(float(len(dataset)) / options.batchSize)))

    #semantic_statistics = []
    #instance_statistics = []
    confusion_matrix = np.zeros((21, 21))
    for sample_index, sample in enumerate(data_iterator):
        if sample_index == options.numTestingImages:
            break

        coords, colors, faces, semantic_gt, instance_gt = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda()

        semantic_pred = semantic_model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])))
        #semantic_pred = semantic_pred.reshape((len(coords), -1))
        semantic_loss = torch.nn.functional.cross_entropy(semantic_pred.view((-1, int(semantic_pred.shape[-1]))), semantic_gt.view(-1), weight=class_weights)
        semantic_pred = semantic_pred.max(-1)[1].unsqueeze(0)

        # print(coords.reshape((-1, 4)).min(0), coords.reshape((-1, 4)).max(0))
        # print(colors.reshape((-1, 3)).min(0), colors.reshape((-1, 3)).max(0))        
        # print(torch.stack([semantic_gt, semantic_pred], dim=-1))
        # exit(1)
        confusion_matrix[semantic_pred.view(-1).detach().cpu().numpy(), semantic_gt.view(-1).detach().cpu().numpy()] += 1
        
        losses = [semantic_loss]
        loss = sum(losses)

        loss_values = [l.data.item() for l in losses]
        epoch_losses.append(loss_values)
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f '%l
            continue
        data_iterator.set_description(status)
        continue
    print('validation loss', np.array(epoch_losses).mean(0))
    confusion_matrix = confusion_matrix / np.maximum(confusion_matrix.sum(-1, keepdims=True), 1)
    print((confusion_matrix * 10).astype(np.int32))
    
    #semantic_statistics = np.array(semantic_statistics).sum(0)
    #print(semantic_statistics[:len(semantic_statistics) // 2].astype(np.float32) / np.maximum(semantic_statistics[len(semantic_statistics) // 2:], 1))
    semantic_model.train()
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

def visualizeBatch(options, coords, faces, colors, num_coords, dicts, indexOffset=0, prefix=''):
    #cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
    #pointColorMap = ColorPalette(20).getColorMap()
    #images = ((images.transpose((0, 2, 3, 1)) + 0.5) * 255).astype(np.uint8)
    for batchIndex in range(len(coords)):
        #cv2.imwrite(filename, image)
        write_ply_color(options.test_dir + '/' + str(indexOffset + batchIndex) + '_input_color.ply', coords[batchIndex], faces[batchIndex], colors[batchIndex][:, :3])
        write_ply_color(options.test_dir + '/' + str(indexOffset + batchIndex) + '_input_normal.ply', coords[batchIndex], faces[batchIndex], colors[batchIndex][:, 3:6])        
        for name, result_dict in dicts:
            semantics = result_dict['semantic'][batchIndex]
            
            filename = options.test_dir + '/' + str(indexOffset + batchIndex) + '_' + name + '_semantic.ply'
            write_ply_label(filename, coords[batchIndex][:len(semantics)], faces[batchIndex], semantics)

            if 'instance' in result_dict:
                instances = result_dict['instance'][batchIndex]

                filename = options.test_dir + '/' + str(indexOffset + batchIndex) + '_' + name + '_instance.ply'                
                write_ply_label(filename, coords[batchIndex][:len(instances)], faces[batchIndex], instances)
                
                instance_semantics = {}
                for instance, semantic in zip(instances, semantics):
                    if instance not in instance_semantics:
                        instance_semantics[instance] = []
                        pass
                    instance_semantics[instance].append(semantic)
                    continue

                semantic_instance_counts = np.zeros(21, dtype=np.int32)                
                for instance, semantic_labels in instance_semantics.items():
                    semantic_labels, counts = np.unique(semantic_labels, return_counts=True)
                    semantic_label = semantic_labels[counts.argmax()]
                    semantic_instance_counts[semantic_label] += 1
                    print(instance, ID_TO_LABEL[semantic_label], counts.max())                    
                    continue          
                for semantic_label, counts in enumerate(semantic_instance_counts):
                    print(ID_TO_LABEL[semantic_label], counts)
                    continue
                
                if False:
                    filename = options.test_dir + '/' + str(indexOffset + batchIndex) + '_' + name + '_edge.ply'                    
                    write_ply_edge(filename, coords[batchIndex], faces[batchIndex], instances)
                    pass
                if name == 'pred':
                    instance_semantic_map = np.zeros(max(list(instance_semantics.keys())) + 1, dtype=np.int32)
                    for instance, semantic_labels in instance_semantics.items():
                        semantic_labels, counts = np.unique(semantic_labels, return_counts=True)
                        instance_semantic_map[instance] = semantic_labels[counts.argmax()]
                        continue
                    print(instance_semantic_map)
                    instance_labels = instance_semantic_map[instances]
                    instance_labels = instance_labels[:num_coords[batchIndex]]
                    labels = np.unique(instances[instance_labels == 6])
                    print(labels)
                    instance_gt = dicts[0][1]['instance'][batchIndex]
                    instance_gt = instance_gt[:num_coords[batchIndex]]                    
                    for label in labels:
                        labels_gt = instance_gt[instances == label]
                        labels_gt, counts = np.unique(labels_gt, return_counts=True)
                        print(np.stack([labels_gt, counts], axis=-1))
                        continue
                    #     filename = options.test_dir + '/' + str(indexOffset + batchIndex) + '_' + name + '_instance_semantic_' + str(label) + '.ply'
                    #     write_ply_label(filename, coords[batchIndex][:num_coords[batchIndex]], faces[batchIndex], instances[:num_coords[batchIndex]], debug_index=label)
                    #     continue
                    
                    filename = options.test_dir + '/' + str(indexOffset + batchIndex) + '_' + name + '_instance_semantic.ply'
                    write_ply_label(filename, coords[batchIndex][:num_coords[batchIndex]], faces[batchIndex], instance_labels[:num_coords[batchIndex]], debug_index=6)
                    pass                
                pass
            #cv2.imwrite(filename.replace('image', info + '_' + name), drawSegmentationImage(result_dict[info][batchIndex], blackIndex=0, blackThreshold=0.5))
            continue
        continue
    return

class InstanceClassifier():
    def __init__(self, full_scale=127, use_normal=False):
        self.full_scale = full_scale
        semantic_model = Classifier(full_scale, use_normal)
        semantic_model.cuda()    
        checkpoint_dir = 'checkpoint/semantics/'
        semantic_model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_semantic.pth'))
        semantic_model.eval()
        self.semantic_model = semantic_model
        return

    def classify(self, coords, colors, instances, semantics):
        unique_instances, first_indices, instances = np.unique(instances, return_index=True, return_inverse=True)
        new_coords = np.zeros(coords.shape, dtype=coords.dtype)
        instance_masks = []
        instance_labels = []
        for instance in range(len(unique_instances)):
            instance_mask = instances == instance
            instance_coords = coords[instance_mask]
            mins = instance_coords.min(0)
            maxs = instance_coords.max(0)
            max_range = (maxs - mins).max()
            padding = max_range * 0.05
            max_range += padding * 2
            mins = (mins + maxs) / 2 - max_range / 2
            instance_coords = np.clip(np.round((instance_coords - mins) / max_range * self.full_scale), 0, self.full_scale - 1)
            new_coords[instance_mask] = instance_coords
            instance_masks.append(instance_mask)
            
            semantic_labels = semantics[instance_mask]
            semantic_labels, counts = np.unique(semantic_labels, return_counts=True)
            instance_labels.append(semantic_labels[counts.argmax()])
            continue
        coords = np.concatenate([new_coords, np.expand_dims(instances, -1)], axis=-1)
        coords = torch.from_numpy(coords.astype(np.int64)).cuda()

        colors = colors.astype(np.float32) / 127.5 - 1        
        colors = torch.from_numpy(colors.astype(np.float32)).cuda()
        semantic_pred = self.semantic_model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])))
        semantic_pred = torch.nn.functional.softmax(semantic_pred, dim=-1)
        confidence, label = semantic_pred.max(-1)

        #print(coords.reshape((-1, 4)).min(0), coords.reshape((-1, 4)).max(0))
        #print(colors.reshape((-1, 3)).min(0), colors.reshape((-1, 3)).max(0))        
        
        label = label.view(-1).detach().cpu().numpy()        
        confidence = confidence.view(-1).detach().cpu().numpy()
        semantic_pred = semantic_pred.detach().cpu().numpy()
        #semantic_pred = semantic_pred[instances]
        #semantics = semantic_pred[first_indices]

        # print(list(zip(semantics[first_indices].tolist(), label.tolist(), (semantic_pred * 10).astype(np.int32).tolist())))
        # exit(1)
        #instance_info = list(zip(instance_masks, label.tolist(), confidence.tolist()))

        if True:
            for info in zip(unique_instances.tolist(), mapper[semantics[first_indices]].tolist(), mapper[label].tolist(), (semantic_pred * 10).astype(np.int32).tolist()):
                print(info)
                continue
            exit(1)
            pass

        confidence = [scores[label] for label, scores in zip(instance_labels, semantic_pred)]
        instance_info = list(zip(instance_masks, instance_labels, confidence))
        #instance_info = list(zip(instance_masks, label.tolist(), confidence.tolist()))
        instance_info = [info for index, info in enumerate(instance_info) if unique_instances[index] >= 0]
        return instance_info
    
if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'semantics'
    #args.keyname += '_' + args.dataset

    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    if args.numScales != 1:
        args.keyname += '_' + str(args.numScales)
        pass    
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = 'test/' + args.keyname

    args.trainingMode = 'semantic'
    args.inputScale = 127
    
    print('keyname=%s task=%s started'%(args.keyname, args.task))

    main(args)
