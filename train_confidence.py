import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import os
import cv2

from utils import *
from options import parse_args

from models.instance import Model, CoordAugmentation, NeighborGT, Validator

from datasets.scannet_dataset import ScanNetDataset

torch.set_printoptions(threshold=5000)


def main(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    confidence_model = Validator(options.inputScale, 'normal' in options.suffix)
    confidence_model.cuda()    
    
    if options.restore == 1:
        print('restore')
        if options.startEpoch >= 0:
            confidence_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_confidence_' + str(options.startEpoch) + '.pth'))
        else:
            confidence_model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint_confidence.pth'))
            pass        
        pass
    
    
    dataset_val = ScanNetDataset(options, split='val', load_confidence=True, random=False)
    if options.task == 'test':
        testOneEpoch(options, confidence_model, dataset_val, validation=False)
        exit(1)
    
    optimizer = torch.optim.Adam(confidence_model.parameters(), lr = options.LR)
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim_confidence.pth'):
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim_confidence.pth'))
        pass

    dataset = ScanNetDataset(options, split='train', load_confidence=True, random=True)
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=True, num_workers=16)

    for epoch in range(options.numEpochs):
        epoch_losses = []
        data_iterator = tqdm(dataloader, total=int(np.ceil(float(len(dataset)) / options.batchSize)))
        for sample_index, sample in enumerate(data_iterator):
            optimizer.zero_grad()
            
            coords, colors, faces, semantic_gt, confidence_gt = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda()

            confidence_pred = confidence_model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])), semantic_gt.view(-1))
            #semantic_pred = semantic_pred.reshape((len(coords), -1))
            confidence_pred = torch.sigmoid(confidence_pred)            
            confidence_loss = torch.nn.functional.binary_cross_entropy(confidence_pred.view(-1), confidence_gt.view(-1).float())

            losses = [confidence_loss]
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
                torch.save(confidence_model.state_dict(), options.checkpoint_dir + '/checkpoint_confidence_' + str(epoch // 10) + '.pth')
                torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_confidence_' + str(epoch // 10) + '.pth')
                pass
            torch.save(confidence_model.state_dict(), options.checkpoint_dir + '/checkpoint_confidence.pth')
            torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_confidence.pth')
            pass
        testOneEpoch(options, confidence_model, dataset_val, validation=True)        
        continue
    return

def testOneEpoch(options, confidence_model, dataset, validation=True):
    for split in ['pred', 'gt']:
        if not os.path.exists(options.test_dir + '/' + split):
            os.system("mkdir -p %s"%options.test_dir + '/' + split)
            pass
        if not os.path.exists(options.test_dir + '/' + split + '/pred_mask'):
            os.system("mkdir -p %s"%options.test_dir + '/' + split + '/pred_mask')
            pass        
        continue

    confidence_model.eval()
    
    # bn = list(list(model.children())[0].children())[3]
    # print(bn.running_var, bn.running_mean)
    # exit(1)
    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=1)
    
    epoch_losses = []    
    data_iterator = tqdm(dataloader, total=int(np.ceil(float(len(dataset)) / options.batchSize)))

    for sample_index, sample in enumerate(data_iterator):
        if sample_index == options.numTestingImages:
            break

        coords, colors, faces, semantic_gt, confidence_gt, instance_masks = sample[0].cuda(), sample[1].cuda(), sample[2].cuda(), sample[3].cuda(), sample[4].cuda(), sample[6]

        confidence_pred = confidence_model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])), semantic_gt.view(-1))
        #semantic_pred = semantic_pred.reshape((len(coords), -1))
        confidence_pred = torch.sigmoid(confidence_pred)            
        confidence_loss = torch.nn.functional.binary_cross_entropy(confidence_pred.view(-1), confidence_gt.view(-1).float())

        losses = [confidence_loss]
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
    confidence_model.train()
    return


class InstanceValidator():
    """ Load trained model to predict confidence for instances """
    def __init__(self, checkpoint_dir, full_scale=127, use_normal=False):
        self.full_scale = full_scale
        confidence_model = Validator(full_scale, use_normal)
        confidence_model.cuda()    
        confidence_model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_confidence.pth'))
        confidence_model.eval()
        self.confidence_model = confidence_model
        return

    def validate(self, coords, colors, instances, semantics):
        instances += 1
        semantic_inp = []        
        instance_masks = []
        new_coords = np.zeros(coords.shape, dtype=coords.dtype)
        confidence_by_counts = []
        for instance in range(instances.max() + 1):
            instance_mask = instances == instance
            if instance_mask.sum() == 0:
                print('sum = 0', instance, instances.max() + 1, instance_mask.sum())
                exit(1)
            info = np.unique(semantics[instance_mask > 0.5], return_counts=True)
            label_pred = info[0][info[1].argmax()]
            instance_coords = coords[instance_mask]
            mins = instance_coords.min(0)
            maxs = instance_coords.max(0)
            max_range = (maxs - mins).max()
            padding = max_range * 0.05
            max_range += padding * 2
            mins = (mins + maxs) / 2 - max_range / 2
            instance_coords = np.clip(np.round((instance_coords - mins) / max_range * self.full_scale), 0, self.full_scale - 1)
            new_coords[instance_mask] = instance_coords

            if instance > 0:
                semantic_inp.append(label_pred)                
                instance_masks.append(instance_mask)
                confidence_by_counts.append(float(info[1].max()) / info[1].sum())                
                pass
            continue
        coords = np.concatenate([new_coords, np.expand_dims(instances, -1)], axis=-1)
        coords = torch.from_numpy(coords.astype(np.int64)).cuda()

        colors = colors.astype(np.float32) / 127.5 - 1        
        colors = torch.from_numpy(colors.astype(np.float32)).cuda()

        semantic_inp = np.stack(semantic_inp).astype(np.int64)
                
        confidence_pred = self.confidence_model(coords.reshape((-1, 4)), colors.reshape((-1, colors.shape[-1])), torch.from_numpy(semantic_inp).view(-1).cuda())
        confidence_pred = torch.sigmoid(confidence_pred)                    
        confidence_pred = confidence_pred.detach().cpu().numpy()
        
        instance_info = list(zip(instance_masks, semantic_inp, confidence_pred))
        
        return instance_info
    
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

    args.suffix = ''
    args.numScales = 0
    args.inputScale = 127
    
    print('keyname=%s task=%s started'%(args.keyname, args.task))

    main(args)
