import sparseconvnet as scn
import torch.nn as nn
import torch
import numpy as np
import time

# class InstanceGT(nn.Module):
#     def __init__(self, full_scale=4096, num_scales=6):
#         nn.Module.__init__(self)        
#         self.full_scale = full_scale
#         self.num_scales = num_scales
#         self.num_instances = 100

#         dimension = 3        
#         self.input_layer = scn.InputLayer(dimension, full_scale, mode=4)
#         self.pool_1 = scn.AveragePooling(dimension, 2, 2)
#         self.pool_2 = scn.AveragePooling(dimension, 4, 4)
#         self.pool_3 = scn.AveragePooling(dimension, 8, 8)
#         self.pool_4 = scn.AveragePooling(dimension, 16, 16)
#         self.pool_5 = scn.AveragePooling(dimension, 32, 32)
#         self.max_pool_1 = scn.MaxPooling(dimension, 2, 2)
#         self.max_pool_2 = scn.MaxPooling(dimension, 4, 4)
#         self.max_pool_3 = scn.MaxPooling(dimension, 8, 8)
#         self.max_pool_4 = scn.MaxPooling(dimension, 16, 16)
#         self.max_pool_5 = scn.MaxPooling(dimension, 32, 32)                                        
#         self.unpool_1 = scn.UnPooling(dimension, 2, 2)
#         self.unpool_2 = scn.UnPooling(dimension, 4, 4)
#         self.unpool_3 = scn.UnPooling(dimension, 8, 8)
#         self.unpool_4 = scn.UnPooling(dimension, 16, 16)
#         self.unpool_5 = scn.UnPooling(dimension, 32, 32)
#         self.unpool_6 = scn.UnPooling(dimension, 64, 64)                                        
#         self.conv_1 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
#         self.conv_2 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)        
#         self.conv_3 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
#         self.conv_4 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
#         self.conv_5 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
#         self.conv_6 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)

#         with torch.no_grad():
#             weights = [torch.zeros(27, self.num_instances, self.num_instances).cuda() for _ in range(6)]
#             for index, offset in enumerate([4, 22, 10, 16, 12, 14]):
#                 for instance_index in range(self.num_instances):
#                     weights[index][offset, instance_index, instance_index] = 1
#                     continue
#                 continue
#             self.conv_1.weight = nn.Parameter(weights[0])
#             self.conv_2.weight = nn.Parameter(weights[1])
#             self.conv_3.weight = nn.Parameter(weights[2])
#             self.conv_4.weight = nn.Parameter(weights[3])
#             self.conv_5.weight = nn.Parameter(weights[4])
#             self.conv_6.weight = nn.Parameter(weights[5])
#             pass
#         self.output_layer = scn.OutputLayer(dimension)
#         return

#     def toSparse(self, coords, neighbors):
#         neighbors = [self.input_layer([coords, neighbor]) for neighbor in neighbors]
#         neighbors = [neighbors[0], self.max_pool_1(neighbors[1]), self.max_pool_2(neighbors[2]), self.max_pool_3(neighbors[3]), self.max_pool_4(neighbors[4]), self.max_pool_5(neighbors[5])]
#         return neighbors

#     def toDense(self, neighbors):
#         neighbors = [neighbors[0], self.unpool_1(neighbors[1]), self.unpool_2(neighbors[2]), self.unpool_3(neighbors[3]), self.unpool_4(neighbors[4]), self.unpool_5(neighbors[5])]
#         neighbors = [self.output_layer(neighbor) for neighbor in neighbors]
#         return neighbors
    

#     def forward(self, coords, instance_gt, use_gpu=True):
#         if use_gpu:
#             instance_gt = self.input_layer((coords, (torch.unsqueeze(instance_gt, -1) == torch.arange(self.num_instances).cuda()).float()))
#         else:
#             instance_gt = self.input_layer((coords, (torch.unsqueeze(instance_gt, -1) == torch.arange(self.num_instances)).float()))
#             pass

#         instances = [instance_gt, self.pool_1(instance_gt), self.pool_2(instance_gt), self.pool_3(instance_gt), self.pool_4(instance_gt), self.pool_5(instance_gt)]
#         for scale, instance in enumerate(instances):
#             #instance.features = instance.features * pow(pow(2, scale), 3)
#             instance.features = instance.features / torch.clamp(torch.sum(instance.features, dim=-1, keepdim=True), min=1e-4)
#             continue
#         # for _ in range(self.num_scales - 1):
#         #     instance_gt = self.avg_pool(instances[-1])
#         #     #instance_gt.features = instance_gt.features / torch.clamp(torch.sum(instance_gt.features, dim=-1, keepdim=True), min=1e-4)
#         #     instances.append(instance_gt)
#         #     continue

#         # instances[2] = self.avg_pool(self.avg_pool(instances[0]))
#         # instances[2] = self.avg_pool_2(instances[0])
        
#         #instances_dense = [instances[0], self.unpool_1(instances[1]), self.unpool_2(instances[2]), self.unpool_3(instances[3]), self.unpool_4(instances[4]), self.unpool_5(instances[5]), self.unpool_6(instances[6])]

#         #print(instances[0].features[3656])
#         #instances_dense = instances
        
#         instances_dense = [instances[0], self.unpool_1(instances[1]), self.unpool_2(instances[2]), self.unpool_3(instances[3]), self.unpool_4(instances[4]), self.unpool_5(instances[5])]

#         instances_dense = [self.output_layer(instance) for instance in instances_dense]

#         # for scale_index, instance in enumerate(instances_dense):
#         #     #for _ in range(scale_index):
#         #     #instance = self.unpool(instance)
#         #     #continue
#         #     instances_dense.append(self.output_layer(instance))
#         #     continue


#         # debug_location = coords[3831][:3].detach().cpu().numpy() // 4
#         # locations = instances[0].get_spatial_locations()[:, :3].detach().cpu().numpy() // 4
#         # features = instances[0].features.detach().cpu().numpy()
#         # debug_features = features[np.all(locations == debug_location, axis=-1)].sum(0)
#         # print(debug_features)
#         # locations = instances[2].get_spatial_locations()[:, :3].detach().cpu().numpy()
#         # features = instances[2].features.detach().cpu().numpy()
#         # debug_features = features[np.all(locations == debug_location, axis=-1)].sum(0)
#         # print(debug_features)
#         # exit(1)        
#         # for location in instances[2].get_spatial_locations()[:, :3].detach().cpu().numpy():
#         #     assert(np.all(locations == location, axis=-1).sum() > 0)
#         #     continue
#         # exit(1)
#         # for scale, instance in enumerate(instances):
#         #     print(instance.features[torch.all(instance.get_spatial_locations()[:, :3] == (coords[3831][:3].cpu() // pow(2, scale)), dim=-1).cpu()])
#         #     continue
#         # exit(1)
#         # print(instances_dense[0].shape, instances_dense[2].shape)
#         # #print((instances_dense[0][3656] * 100).int())
#         # print((instances_dense[0][3831] * 100).int())                    
#         # #print((instances_dense[2][3656] * 100).int())                    
#         # #print((instances_dense[2][3867] * 100).int())
#         # print((instances_dense[1][3831] * 100).int())                
#         # print((instances_dense[2][3831] * 100).int())
#         # print((instances_dense[3][3831] * 100).int())
#         # print((instances_dense[4][3831] * 100).int())                        
#         # exit(1)
        
#         neighbors = [[] for _ in range(self.num_scales)]
#         for scale_index, instance in enumerate(instances):
#             shifted_instances = [self.conv_1(instance), self.conv_2(instance), self.conv_3(instance), self.conv_4(instance), self.conv_5(instance), self.conv_6(instance)]
#             for shifted_instance in shifted_instances:
#                 if scale_index == 1:
#                     shifted_instance = self.unpool_1(shifted_instance)
#                 elif scale_index == 2:
#                     shifted_instance = self.unpool_2(shifted_instance)
#                 elif scale_index == 3:
#                     shifted_instance = self.unpool_3(shifted_instance)
#                 elif scale_index == 4:
#                     shifted_instance = self.unpool_4(shifted_instance)
#                 elif scale_index == 5:
#                     shifted_instance = self.unpool_5(shifted_instance)
#                 elif scale_index == 6:
#                     shifted_instance = self.unpool_6(shifted_instance)
#                     pass
#                 shifted_instance = self.output_layer(shifted_instance)
#                 for prev_scale_index, prev_instance in enumerate(instances_dense[:scale_index + 1]):
#                     matching = torch.sum(shifted_instance * prev_instance, dim=-1)
#                     neighbors[prev_scale_index].append(matching)
#                     continue
#                 pass
#             for prev_scale_index, prev_instance in enumerate(instances_dense[:scale_index]):
#                 matching = torch.sum(instances_dense[scale_index] * prev_instance, dim=-1)
#                 neighbors[prev_scale_index].append(matching)
#                 continue
#             continue
#         neighbors = [torch.stack(neighbor, dim=-1) for neighbor in neighbors]
#         neighbors = self.toSparse(coords, neighbors)
#         #print(neighbors[1].get_spatial_locations, neighbors[1].features.shape, neighbors[1].features.sum(-1).min())
#         return neighbors

class InstanceGTScale(nn.Module):
    def __init__(self, options):
        nn.Module.__init__(self)
        self.options = options
        self.num_instances = 100
        dimension = 3        
        self.input_layer = scn.InputLayer(dimension, options.inputScale, mode=4)
        self.conv_1 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
        self.conv_2 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)        
        self.conv_3 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
        self.conv_4 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
        self.conv_5 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
        self.conv_6 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)

        self.pool_1 = scn.AveragePooling(dimension, 2, 2)
        self.pool_2 = scn.AveragePooling(dimension, 4, 4)
        self.pool_3 = scn.AveragePooling(dimension, 8, 8)
        self.pool_4 = scn.AveragePooling(dimension, 16, 16)
        self.pool_5 = scn.AveragePooling(dimension, 32, 32)
        self.unpool_1 = scn.UnPooling(dimension, 2, 2)
        self.unpool_2 = scn.UnPooling(dimension, 4, 4)
        self.unpool_3 = scn.UnPooling(dimension, 8, 8)
        self.unpool_4 = scn.UnPooling(dimension, 16, 16)
        self.unpool_5 = scn.UnPooling(dimension, 32, 32)
        self.unpool_6 = scn.UnPooling(dimension, 64, 64)
        
        with torch.no_grad():
            weights = [torch.zeros(27, self.num_instances, self.num_instances).cuda() for _ in range(6)]
            for index, offset in enumerate([4, 22, 10, 16, 12, 14]):
                for instance_index in range(self.num_instances):
                    weights[index][offset, instance_index, instance_index] = 1
                    continue
                continue
            self.conv_1.weight = nn.Parameter(weights[0])
            self.conv_2.weight = nn.Parameter(weights[1])
            self.conv_3.weight = nn.Parameter(weights[2])
            self.conv_4.weight = nn.Parameter(weights[3])
            self.conv_5.weight = nn.Parameter(weights[4])
            self.conv_6.weight = nn.Parameter(weights[5])
            pass
        self.output_layer = scn.OutputLayer(dimension)
        return

    def toDense(self, neighbors):
        if len(neighbors) >= 2:
            neighbors[1] = self.unpool_1(neighbors[1])
            pass
        if len(neighbors) >= 3:
            neighbors[2] = self.unpool_2(neighbors[2])
            pass
        if len(neighbors) >= 4:
            neighbors[3] = self.unpool_3(neighbors[3])
            pass
        if len(neighbors) >= 5:
            neighbors[4] = self.unpool_4(neighbors[4])
            pass
        if len(neighbors) >= 6:
            neighbors[5] = self.unpool_5(neighbors[5])
            pass
        
        for neighbor in neighbors:
            #neighbor.features = neighbor.features[:, :neighbor.shape[-1] // 2] * neighbor.features[:, neighbor.shape[-1] // 2:]
            neighbor.features = neighbor.features[:, :6]
            continue

        #print(neighbor.features[torch.all(neighbor.get_spatial_locations()[:, :3] == (torch.Tensor([7, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])
        #print(neighbor.features[torch.all(neighbor.get_spatial_locations()[:, :3] == (torch.Tensor([8, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])        
        neighbors = [self.output_layer(neighbor) for neighbor in neighbors]
        return neighbors
    
    def forward(self, coords, instance_gt, use_gpu=True):
        if use_gpu:
            instance = self.input_layer((coords, (torch.unsqueeze(instance_gt, -1) == torch.arange(self.num_instances).cuda()).float()))
        else:
            instance = self.input_layer((coords, (torch.unsqueeze(instance_gt, -1) == torch.arange(self.num_instances)).float()))
            pass

        instances = []
        if self.options.numScales >= 1:
            instances.append(instance)
            pass
        if self.options.numScales >= 2:
            instances.append(self.pool_1(instance))
            pass
        if self.options.numScales >= 3:
            instances.append(self.pool_2(instance))
            pass
        if self.options.numScales >= 4:
            instances.append(self.pool_3(instance))
            pass
        if self.options.numScales >= 5:
            instances.append(self.pool_4(instance))
            pass
        if self.options.numScales >= 6:
            instances.append(self.pool_5(instance))
            pass

        # for instance in instances:
        #     instance.features = instance.features / torch.clamp(torch.sum(instance.features, dim=-1, keepdim=True), min=1e-4)
        #     continue

        # for scale, instance in enumerate(instances):
        #     print(instance.features[torch.all(instance.get_spatial_locations()[:, :3] == (torch.Tensor([7, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])
        #     print(instance.features[torch.all(instance.get_spatial_locations()[:, :3] == (torch.Tensor([8, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])
        #     continue

        scale_count_thresholds = pow(4.0, np.arange(self.options.numScales)) / pow(8, np.arange(self.options.numScales))
        
        for scale, instance in enumerate(instances):
            neighbors = []

            feature_counts = torch.sum(instance.features, dim=-1, keepdim=True)
            instance.features = instance.features / torch.clamp(feature_counts, min=1e-4)
            
            shifted_instances = [self.conv_1(instance), self.conv_2(instance), self.conv_3(instance), self.conv_4(instance), self.conv_5(instance), self.conv_6(instance)]
            features = []
            masks = []
            for shifted_instance in shifted_instances:
                features.append(torch.sum(shifted_instance.features * instance.features, dim=-1))
                masks.append(shifted_instance.features.sum(-1) > 0)
                continue
            features = torch.stack(features, dim=-1)
            masks = torch.stack(masks, dim=-1)
            if scale >= 1:
                masks = masks & (feature_counts >= scale_count_thresholds[scale])
                pass
            features = (features > 0.95).float()
            instance.features = torch.cat([features.detach(), masks.float().detach()], dim=-1)
            continue

        # for scale, instance in enumerate(instances):
        #     print(instance.features[torch.all(instance.get_spatial_locations()[:, :3] == (torch.Tensor([7, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])
        #     print(instance.features[torch.all(instance.get_spatial_locations()[:, :3] == (torch.Tensor([8, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])
        #     continue
        
        return instances


class NeighborGT(nn.Module):
    def __init__(self, options):
        nn.Module.__init__(self)
        self.options = options
        dimension = 3        
        self.input_layer = scn.InputLayer(dimension, options.inputScale, mode=4)
        self.conv = scn.SubmanifoldConvolution(dimension, 1, options.numNeighbors, 3, bias=False)

        self.pool_1 = scn.AveragePooling(dimension, 2, 2)
        self.pool_2 = scn.AveragePooling(dimension, 4, 4)
        self.pool_3 = scn.AveragePooling(dimension, 8, 8)
        self.pool_4 = scn.AveragePooling(dimension, 16, 16)
        self.pool_5 = scn.AveragePooling(dimension, 32, 32)
        self.unpool_1 = scn.UnPooling(dimension, 2, 2)
        self.unpool_2 = scn.UnPooling(dimension, 4, 4)
        self.unpool_3 = scn.UnPooling(dimension, 8, 8)
        self.unpool_4 = scn.UnPooling(dimension, 16, 16)
        self.unpool_5 = scn.UnPooling(dimension, 32, 32)
        
        with torch.no_grad():
            weight = torch.zeros(27, 1, options.numNeighbors).cuda()
            if options.numNeighbors == 6:
                offsets = [4, 22, 10, 16, 12, 14]
                pass
            for index, offset in enumerate(offsets):
                weight[offset, 0, index] = 1
                continue
            self.conv.weight = nn.Parameter(weight)
            pass
        self.output_layer = scn.OutputLayer(dimension)
        return

    def toDense(self, neighbors):
        if len(neighbors) >= 2:
            neighbors[1] = self.unpool_1(neighbors[1])
            pass
        if len(neighbors) >= 3:
            neighbors[2] = self.unpool_2(neighbors[2])
            pass
        if len(neighbors) >= 4:
            neighbors[3] = self.unpool_3(neighbors[3])
            pass
        if len(neighbors) >= 5:
            neighbors[4] = self.unpool_4(neighbors[4])
            pass
        if len(neighbors) >= 6:
            neighbors[5] = self.unpool_5(neighbors[5])
            pass
        
        for neighbor in neighbors:
            neighbor.features = neighbor.features[:, :6]
            continue

        #print(neighbor.features[torch.all(neighbor.get_spatial_locations()[:, :3] == (torch.Tensor([7, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])
        #print(neighbor.features[torch.all(neighbor.get_spatial_locations()[:, :3] == (torch.Tensor([8, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])        
        neighbors = [self.output_layer(neighbor) for neighbor in neighbors]
        return neighbors
    
    def forward(self, coords, instance_gt, use_gpu=True):
        #instance = self.input_layer((coords, torch.unsqueeze(instance_gt + 1, -1).float()))
        instance = self.input_layer((coords, torch.stack([instance_gt.float() + 1, torch.ones(len(instance_gt)).cuda()], dim=-1)))
        instances = []
        if self.options.numScales >= 1:
            instances.append(instance)
            pass
        if self.options.numScales >= 2:
            instances.append(self.pool_1(instance))
            pass
        if self.options.numScales >= 3:
            instances.append(self.pool_2(instance))
            pass
        if self.options.numScales >= 4:
            instances.append(self.pool_3(instance))
            pass
        if self.options.numScales >= 5:
            instances.append(self.pool_4(instance))
            pass
        if self.options.numScales >= 6:
            instances.append(self.pool_5(instance))
            pass

        # for instance in instances:
        #     instance.features = instance.features / torch.clamp(torch.sum(instance.features, dim=-1, keepdim=True), min=1e-4)
        #     continue

        # for scale, instance in enumerate(instances):
        #     print(instance.features[torch.all(instance.get_spatial_locations()[:, :3] == (torch.Tensor([7, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])
        #     print(instance.features[torch.all(instance.get_spatial_locations()[:, :3] == (torch.Tensor([8, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])
        #     continue

        instance_counts = [instance.features[:, 1:2] for instance in instances]
        for instance, count in zip(instances, instance_counts):
            instance.features = instance.features[:, :1] / count
            continue
        scale_count_thresholds = pow(4.0, np.arange(self.options.numScales)) / pow(8, np.arange(self.options.numScales))
        
        for scale, instance in enumerate(instances):
            neighbors = []

            #feature_counts = torch.sum(instance.features, dim=-1, keepdim=True)
            #instance.features = instance.features / torch.clamp(feature_counts, min=1e-4)            
            shifted_instances = self.conv(instance)
            features = ((shifted_instances.features - instance.features).abs() < 0.01).float()
            masks = shifted_instances.features > 0
            # if scale >= 1:
            #     masks = masks & (feature_counts >= scale_count_thresholds[scale])
            #     pass
            instance.features = torch.cat([features.detach(), masks.float().detach()], dim=-1)
            continue

        # for scale, instance in enumerate(instances):
        #     print(instance.features[torch.all(instance.get_spatial_locations()[:, :3] == (torch.Tensor([7, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])
        #     print(instance.features[torch.all(instance.get_spatial_locations()[:, :3] == (torch.Tensor([8, 8, 24]).long() // pow(2, scale)), dim=-1).cpu()])
        #     continue
        return instances


class CoordAugmentation(nn.Module):
    def __init__(self, options):
        nn.Module.__init__(self)
        self.options = options
        dimension = 3
        self.input_layer = scn.InputLayer(dimension, options.inputScale, mode=4)
        
        self.coord_input_layer = scn.InputLayer(dimension, options.inputScale, mode=2)
        self.num_instances = 1
        self.conv_1 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
        self.conv_2 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)        
        self.conv_3 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
        self.conv_4 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
        self.conv_5 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)
        self.conv_6 = scn.SubmanifoldConvolution(dimension, self.num_instances, self.num_instances, 3, bias=False)

        with torch.no_grad():
            weights = [torch.zeros(27, self.num_instances, self.num_instances).cuda() for _ in range(6)]
            for index, offset in enumerate([4, 22, 10, 16, 12, 14]):
                for instance_index in range(self.num_instances):
                    weights[index][offset, instance_index, instance_index] = 1
                    continue
                continue
            self.conv_1.weight = nn.Parameter(weights[0])
            self.conv_2.weight = nn.Parameter(weights[1])
            self.conv_3.weight = nn.Parameter(weights[2])
            self.conv_4.weight = nn.Parameter(weights[3])
            self.conv_5.weight = nn.Parameter(weights[4])
            self.conv_6.weight = nn.Parameter(weights[5])
            pass
        self.coord_output_layer = scn.OutputLayer(dimension)
        return

    def eval(self):
        torch.manual_seed(0)
        return
    
    def train(self):
        t = int(time.time() * 1000000)
        seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
        torch.manual_seed(seed)
        return
    
    def forward(self, coords, faces, colors, instances):
        edges_1 = coords[faces[:, 0]] - coords[faces[:, 1]]
        edges_2 = coords[faces[:, 1]] - coords[faces[:, 2]]
        edges_3 = coords[faces[:, 2]] - coords[faces[:, 0]]
        edge_lengths = torch.max(torch.max(edges_1.abs().max(dim=-1)[0], edges_2.abs().max(dim=-1)[0]), edges_3.abs().max(dim=-1)[0])
        large_faces = faces[edge_lengths >= 2]
        if len(large_faces) == 0:
            return coords[0:1]
        #cross_product = torch.stack([edges_1[:, 1] * edges_2[:, 2] - edges_1[:, 2] * edges_2[:, 1], edges_1[:, 2] * edges_2[:, 0] - edges_1[:, 0] * edges_2[:, 2], edges_1[:, 0] * edges_2[:, 1] - edges_1[:, 1] * edges_2[:, 0]], axis=-1)
        num_sampled_points = 5
        #torch.backends.cudnn.deterministic=True
        alphas = torch.rand(num_sampled_points, len(large_faces), 3).cuda()
        alphas = alphas / alphas.sum(-1, keepdim=True)
        vertices = torch.stack([coords[large_faces[:, 0]], coords[large_faces[:, 1]], coords[large_faces[:, 2]]], dim=1).float()
        sampled_coords = (alphas.unsqueeze(-1) * vertices).sum(2).view((-1, 4)).round().long()
        all_coords = torch.cat([coords, sampled_coords], dim=0)
        
        vertex_colors = torch.stack([colors[large_faces[:, 0]], colors[large_faces[:, 1]], colors[large_faces[:, 2]]], dim=1).float()
        sampled_colors = (alphas.unsqueeze(-1) * vertex_colors).sum(2).view((-1, colors.shape[-1]))
        all_colors = torch.cat([colors, sampled_colors], dim=0)
        
        vertex_instances = torch.stack([instances[large_faces[:, 0]], instances[large_faces[:, 1]], instances[large_faces[:, 2]]], dim=1).float()
        sampled_instances = vertex_instances[torch.arange(len(vertex_instances)).cuda().repeat(num_sampled_points), alphas.max(-1)[1].view(-1)]
        all_instances = torch.cat([instances.float(), sampled_instances], dim=0)

        all_flags = torch.cat([-100 * torch.ones((len(coords), 1)).cuda(), torch.ones((len(sampled_coords), 1)).cuda()], dim=0)

        all_values = torch.cat([all_colors, all_instances, all_flags], dim=-1)

        voxel = self.input_layer((all_coords, all_values))
        
        valid_mask = voxel.features[:, -1] > 0
        valid_values = voxel.features[valid_mask]

        augmented_coords = voxel.get_spatial_locations()[valid_mask].cuda()

        all_coords = torch.cat([coords, augmented_coords], dim=0)
        #augmented_coord_indices = torch.arange(len(coords), len(coords) + len(augmented_coords)).cuda()
        all_coord_indices = torch.arange(1, 1 + len(all_coords)).cuda()        
        coord_indices = self.coord_input_layer((all_coords, all_coord_indices.unsqueeze(-1).float()))
        neighbors = []
        shifted_coord_indices = [self.conv_1(coord_indices), self.conv_2(coord_indices), self.conv_3(coord_indices), self.conv_4(coord_indices), self.conv_5(coord_indices), self.conv_6(coord_indices)]
        shifted_coord_indices = [self.coord_output_layer(indices) for indices in shifted_coord_indices]
        shifted_coord_indices = torch.cat(shifted_coord_indices, dim=0).long().view(-1)
        all_edges = torch.stack([all_coord_indices.repeat(6), shifted_coord_indices], dim=-1)
        augmented_edges = all_edges[(shifted_coord_indices > 0) & (shifted_coord_indices.max(-1)[0] > len(coords))] - 1
        return augmented_coords, valid_values[:, :colors.shape[-1]], valid_values[:, colors.shape[-1]:-1].round().long(), augmented_edges
        
class Model(nn.Module):
    def __init__(self, options):
        nn.Module.__init__(self)

        self.options = options
        
        dimension = 3
        m = 32 # 16 or 32
        residual_blocks = True #True or False
        block_reps = 2 #Conv block repetition factor: 1 or 2

        self.outputs = []
        def hook(module, input, output):
            self.outputs.append(output)
            return
    
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, options.inputScale, mode=4)).add(
            scn.SubmanifoldConvolution(dimension, 3 + 3 * int('normal' in self.options.suffix), m, 3, False)).add(
            scn.UNet(dimension, block_reps, [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], residual_blocks)).add(
            scn.BatchNormReLU(m))
        #print(self.sparseModel[2])
        #exit(1)

        if options.numScales >= 2:
            #list(self.sparseModel[2][1].children())[1][2][3].register_forward_hook(hook)
            list(self.sparseModel[2][4].children())[1][3].register_forward_hook(hook)
            pass
        if options.numScales >= 3:
            list(list(self.sparseModel[2][4].children())[1][2][4].children())[1][3].register_forward_hook(hook)
            pass
        if options.numScales >= 4:
            list(list(list(self.sparseModel[2][4].children())[1][2][4].children())[1][2][4].children())[1][3].register_forward_hook(hook)
            pass        
        if options.numScales >= 5:
            list(list(list(list(self.sparseModel[2][4].children())[1][2][4].children())[1][2][4].children())[1][2][4].children())[1][3].register_forward_hook(hook)            
            pass        
        if options.numScales >= 6:
            list(list(list(list(list(self.sparseModel[2][4].children())[1][2][4].children())[1][2][4].children())[1][2][4].children())[1][2][4].children())[1][3].register_forward_hook(hook)            
            pass        
        
        # list(list(list(list(list(self.sparseModel[2][1].children())[1][2][1].children())[1][2][1].children())[1][2][1].children())[1][2][1].children())[1][2][3].register_forward_hook(hook)
        # list(list(list(list(self.sparseModel[2][1].children())[1][2][1].children())[1][2][1].children())[1][2][1].children())[1][2][3].register_forward_hook(hook)
        # list(list(list(self.sparseModel[2][1].children())[1][2][1].children())[1][2][1].children())[1][2][3].register_forward_hook(hook)
        # list(list(self.sparseModel[2][1].children())[1][2][1].children())[1][2][3].register_forward_hook(hook)        
        # list(self.sparseModel[2][1].children())[1][2][3].register_forward_hook(hook)

        self.sparsify = scn.Sparsify(dimension)
        self.output_layer = scn.OutputLayer(dimension)

        self.linear = nn.Linear(m, 20)
        
        self.neighbor_linear_0 = nn.Linear(m, 6 + 7 * min(self.options.numCrossScales, max(self.options.numScales - 1, 0)))
        self.neighbor_linear_1 = nn.Linear(m * 2, 6 + 7 * min(self.options.numCrossScales, max(self.options.numScales - 2, 0)))
        self.neighbor_linear_2 = nn.Linear(m * 3, 6 + 7 * min(self.options.numCrossScales, max(self.options.numScales - 3, 0)))
        self.neighbor_linear_3 = nn.Linear(m * 4, 6 + 7 * min(self.options.numCrossScales, max(self.options.numScales - 4, 0)))
        self.neighbor_linear_4 = nn.Linear(m * 5, 6 + 7 * min(self.options.numCrossScales, max(self.options.numScales - 5, 0)))
        self.neighbor_linear_5 = nn.Linear(m * 6, 6 + 7 * min(self.options.numCrossScales, max(self.options.numScales - 6, 0)))

        # self.neighbor_linear_0 = nn.Linear(m, 6 + 7 * 5)
        # self.neighbor_linear_1 = nn.Linear(m * 2, 6 + 7 * 4)
        # self.neighbor_linear_2 = nn.Linear(m * 3, 6 + 7 * 3)
        # self.neighbor_linear_3 = nn.Linear(m * 4, 6 + 7 * 2)
        # self.neighbor_linear_4 = nn.Linear(m * 5, 6 + 7 * 1)
        # self.neighbor_linear_5 = nn.Linear(m * 6, 6 + 7 * 0)
        
        self.unpool_1 = scn.UnPooling(dimension, 2, 2)
        self.unpool_2 = scn.UnPooling(dimension, 4, 4)
        self.unpool_3 = scn.UnPooling(dimension, 8, 8)
        self.unpool_4 = scn.UnPooling(dimension, 16, 16)
        self.unpool_5 = scn.UnPooling(dimension, 32, 32)        
        return
    
    def forward(self, coords, colors):
        # for scale_index, neighbor in enumerate(neighbors):
        #     for index, _ in enumerate(neighbor):
        #         print(scale_index, index, _.shape, _.min().item(), _.max().item())
        #         continue
        #     continue
        # exit(1)
        x = self.sparseModel((coords, colors))
        # print(x.shape, len(self.outputs))
        # exit(1)
        if 'maxpool' not in self.options.suffix:
            semantic_pred = self.linear(self.output_layer(x))
        else:
            semantic_pred = self.output_layer(x)
            pass
        
        if False:
            self.outputs = self.outputs[-5:]        
            outputs = [x] + self.outputs[::-1]
            outputs[0].features, outputs[1].features, outputs[2].features, outputs[3].features, outputs[4].features, outputs[5].features = self.neighbor_linear_0(outputs[0].features), self.neighbor_linear_1(outputs[1].features), self.neighbor_linear_2(outputs[2].features), self.neighbor_linear_3(outputs[3].features), self.neighbor_linear_4(outputs[4].features), self.neighbor_linear_5(outputs[5].features)
            neighbor_pred = outputs
        else:
            if self.options.numScales > 1:
                self.outputs = self.outputs[-(self.options.numScales - 1):]
                outputs = [x] + self.outputs[::-1]
            elif self.options.numScales == 1:
                outputs = [x]
            else:
                outputs = []                
                pass

            if self.options.numScales >= 1:
                outputs[0].features = self.neighbor_linear_0(outputs[0].features)[:, :6 + min(self.options.numCrossScales, self.options.numScales - 1) * 7]
                pass
            if self.options.numScales >= 2:
                outputs[1].features = self.neighbor_linear_1(outputs[1].features)[:, :6 + min(self.options.numCrossScales, self.options.numScales - 2) * 7]
                pass
            if self.options.numScales >= 3:
                outputs[2].features = self.neighbor_linear_2(outputs[2].features)[:, :6 + min(self.options.numCrossScales, self.options.numScales - 3) * 7]
                pass
            if self.options.numScales >= 4:
                outputs[3].features = self.neighbor_linear_3(outputs[3].features)[:, :6 + min(self.options.numCrossScales, self.options.numScales - 4) * 7]
                pass
            if self.options.numScales >= 5:
                outputs[4].features = self.neighbor_linear_4(outputs[4].features)[:, :6 + min(self.options.numCrossScales, self.options.numScales - 5) * 7]
                pass
            if self.options.numScales >= 6:
                outputs[5].features = self.neighbor_linear_5(outputs[5].features)[:, :6 + min(self.options.numCrossScales, self.options.numScales - 6) * 7]
                pass                                    
            neighbor_pred = outputs
            pass
        
        #neighbor_pred = self.neighbor_linear_0(outputs[0].features), self.neighbor_linear_1(outputs[1].features), self.neighbor_linear_2(outputs[2].features), self.neighbor_linear_3(outputs[3].features), self.neighbor_linear_4(outputs[4].features), self.neighbor_linear_5(outputs[5].features)

        # outputs = [self.unpool_1(outputs[0]), self.unpool_2(outputs[1]), self.unpool_3(outputs[2]), self.unpool_4(outputs[3]), self.unpool_5(outputs[4])]
        # outputs = [self.output_layer(output) for output in outputs]
        # outputs = [x] + outputs
        # neighbor_pred = [self.neighbor_linear_0(outputs[0]), self.neighbor_linear_1(outputs[1]), self.neighbor_linear_2(outputs[2]), self.neighbor_linear_3(outputs[3]), self.neighbor_linear_4(outputs[4]), self.neighbor_linear_5(outputs[5])]
        
        #neighbor_pred_0 = self.neighbor_linear_0(x)
        return semantic_pred, neighbor_pred

class SemanticClassifier(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.linear = nn.Linear(32, 21)
        self.invalid_label = torch.Tensor([20]).cuda().long()
        return

    def forward(self, features, instances, invalid_instances=[], semantics=[]):
        new_features = []
        semantic_gt = []
        instance_masks = []        
        for instance in range(1, instances.max() + 1):
            instance_mask = instances == instance
            indices = torch.nonzero(instance_mask)
            if len(indices) < 100:
                continue
            if len(semantics) == 0:
                semantic_gt.append(self.invalid_label)
            else:
                semantic_gt.append(semantics[indices[0]])
                pass
            new_features.append(torch.max(features[instance_mask], dim=0)[0])
            instance_masks.append(instance_mask)
            continue

        if len(invalid_instances) > 0:
            for instance in [np.random.randint(1, invalid_instances.max() + 1)]:
            #for instance in range(1, invalid_instances.max() + 1):
                instance_mask = invalid_instances == instance
                if instance_mask.sum() < 100:
                    continue
                semantic_gt.append(self.invalid_label)
                new_features.append(torch.max(features[instance_mask], dim=0)[0])
                continue
            pass
        new_features = torch.stack(new_features, dim=0)
        semantic_gt = torch.stack(semantic_gt, dim=0).view(-1).detach()
        semantic_pred = self.linear(new_features)
        #semantic_pred = semantic_pred[instances]
        return semantic_pred, semantic_gt, instance_masks
        

class Classifier(nn.Module):
    def __init__(self, full_scale=127, use_normal=False):
        nn.Module.__init__(self)

        dimension = 3
        m = 32 # 16 or 32
        residual_blocks = True #True or False
        block_reps = 2 #Conv block repetition factor: 1 or 2

        blocks = [['b', m * k, 2, 2] for k in [1, 2, 3, 4, 5]]
        self.num_final_channels = m * len(blocks)

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(dimension, 3 + 3 * int(use_normal), m, 3, False)).add(
            scn.MaxPooling(dimension, 3, 2)).add(
            scn.SparseResNet(dimension, m, blocks)).add(
            scn.BatchNormReLU(self.num_final_channels)).add(
            scn.SparseToDense(dimension, self.num_final_channels))

        self.pred = nn.Linear(m * len(blocks), 21)
        return
    
    def forward(self, coords, colors):
        x = self.sparseModel((coords, colors))
        pred = self.pred(x.view((-1, self.num_final_channels)))
        return pred

class Validator(nn.Module):
    def __init__(self, full_scale=127, use_normal=False):
        nn.Module.__init__(self)

        dimension = 3
        m = 32 # 16 or 32
        residual_blocks = True #True or False
        block_reps = 2 #Conv block repetition factor: 1 or 2

        blocks = [['b', m * k, 2, 2] for k in [1, 2, 3, 4, 5]]
        self.num_final_channels = m * len(blocks)

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(dimension, 3 + 3 * int(use_normal), m, 3, False)).add(
            scn.MaxPooling(dimension, 3, 2)).add(
            scn.SparseResNet(dimension, m, blocks)).add(
            scn.BatchNormReLU(self.num_final_channels)).add(
            scn.SparseToDense(dimension, self.num_final_channels))

        self.num_labels = 20
        self.label_encoder = nn.Sequential(nn.Linear(self.num_labels, 64), nn.ReLU())
        #self.pred = nn.Linear(m * len(blocks), 1)
        self.pred = nn.Sequential(nn.Linear(self.num_final_channels + 64, 64), nn.ReLU(), nn.Linear(64, 1))
        return
    
    def forward(self, coords, colors, labels):
        x = self.sparseModel((coords, colors))
        labels = (labels.unsqueeze(-1) == torch.arange(self.num_labels).cuda()).float()
        label_x = self.label_encoder(labels)
        x = torch.cat([x.view((-1, self.num_final_channels))[1:], label_x], dim=1)
        pred = self.pred(x)
        return pred.view(-1)

class FittingModel(nn.Module):
    def __init__(self, full_scale=255, use_normal=False):
        nn.Module.__init__(self)

        self.full_scale = full_scale
        dimension = 3
        m = 32 # 16 or 32
        residual_blocks = True #True or False
        block_reps = 2 #Conv block repetition factor: 1 or 2

        blocks = [['b', m * k, 2, 2] for k in [1, 2, 3, 4, 5, 6]]
        self.num_final_channels = m * len(blocks)

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(dimension, 3 + 3 * int(use_normal), m, 3, False)).add(
            scn.MaxPooling(dimension, 3, 2)).add(
            scn.SparseResNet(dimension, m, blocks)).add(
            scn.BatchNormReLU(self.num_final_channels)).add(
            scn.SparseToDense(dimension, self.num_final_channels))

        self.pred = nn.Linear(m * len(blocks), 6)
        return
    
    def forward(self, coords, colors):
        x = self.sparseModel((coords, colors))
        parameter_pred = self.pred(x.view((-1, self.num_final_channels)))
        tangent_pred = parameter_pred[:, :2]
        tangent_pred = tangent_pred / torch.norm(tangent_pred, dim=-1, keepdim=True)
        center_pred = parameter_pred[:, 2:5] * self.full_scale
        size_pred = parameter_pred[:, 5:7] * self.full_scale

        num_sampled_points = 1000
        alphas = torch.rand(num_sampled_points, 2).cuda()
        UVs = (alphas - 0.5) * size_pred.unsqueeze(1)
        XYZ = torch.cat([UVs[:, :, 0:1] * tangent_pred.unsqueeze(1), UVs[:, :, 1:2]], dim=-1) + center.unsqueeze(1)
        XYZ = torch.clamp(torch.round(XYZ).int(), min=0, max=full_scale-1)
        return pred
    
