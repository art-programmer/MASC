# MASC: Multi-scale Affinity with Sparse Convolution for 3D Instance Segmentation (Technical Report)
## Introduction
This is the PyTorch implementation for our [technical report]() which achieves the state-of-the-art performance on the 3D instance segmentation task of the ScanNet benchmark.

## Installation
```
pip install -r requirements.txt
```

## Data preparation
To prepare training data from ScanNet mesh models, please run:
```
python train.py --task=prepare --dataFolder=[SCANNET_PATH] ----labelFile=[SCANNET_LABEL_FILE_PATH (i.e., scannetv2-labels.combined.tsv)]
```

## Training
To train the main model which predict semantics and affinities, please run:
```
python train.py --restore=0 --dataFolder=[SCANNET_PATH]
```

## Validation
To validate the trained model, please run:
```
python train.py --restore=1 --dataFolder=[SCANNET_PATH] --task=test
```

## Inference
To run the inference using the trained model, please run:

```
python inference.py --restore=0 --dataFolder=[SCANNET_PATH] --task=predict_cluster split=val
```
The task option indicates:
- "predict": predict semantics and affinities
- "cluster": run the clustering algorithm based on the predicted affinities
- "write": write instance segmentation results
The task option can contain any combinations of these three tasks, but the earlier task must be run before later tasks.

## Write results for the final evaluation
To train the instance confidence model, please first generate the instance segmentation results:
```
python inference.py --restore=0 --dataFolder=[SCANNET_PATH] --task=predict_cluster split=val
python inference.py --restore=0 --dataFolder=[SCANNET_PATH] --task=predict_cluster split=train
```
Then train the confidence model:
```
python train.py --restore=0 --dataFolder=[SCANNET_PATH]
```
Predict instance confidence, add additional instances for certain semantic labels, and write instance segmentation results:
```
python inference.py --restore=0 --dataFolder=[SCANNET_PATH] --task=predict_cluster_write split=test
```
