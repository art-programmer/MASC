import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='PlaneFlow')
    
    parser.add_argument('--task', dest='task',
                        help='task type: [train, test, predict]',
                        default='train', type=str)
    parser.add_argument('--restore', dest='restore',
                        help='how to restore the model',
                        default=1, type=int)
    parser.add_argument('--batchSize', dest='batchSize',
                        help='batch size',
                        default=1, type=int)
    parser.add_argument('--numTrainingImages', dest='numTrainingImages',
                        help='the number of images to train',
                        default=0, type=int)
    parser.add_argument('--numTestingImages', dest='numTestingImages',
                        help='the number of images to test/predict',
                        default=20, type=int)
    parser.add_argument('--scene_id', dest='scene_id',
                        help='scene id to test',
                        default='', type=str)
    parser.add_argument('--LR', dest='LR',
                        help='learning rate',
                        default=2.5e-4, type=float)
    parser.add_argument('--numEpochs', dest='numEpochs',
                        help='the number of epochs',
                        default=50, type=int)
    parser.add_argument('--startEpoch', dest='startEpoch',
                        help='starting epoch index',
                        default=-1, type=int)
    parser.add_argument('--inputScale', dest='inputScale',
                        help='input scale',
                        default=4096, type=int)
    parser.add_argument('--scanScale', dest='scanScale',
                        help='scan scale',
                        default=50, type=int)    
    parser.add_argument('--numScales', dest='numScales',
                        help='the number of scales',
                        default=2, type=int)
    parser.add_argument('--numCrossScales', dest='numCrossScales',
                        help='the number of cross scales',
                        default=0, type=int)            
    parser.add_argument('--numNeighbors', dest='numNeighbors',
                        help='the number of neighbors',
                        default=6, type=int)
    parser.add_argument('--negativeWeights', dest='negativeWeights',
                        help='negative weights',
                        default='531111', type=str)
    parser.add_argument('--visualizeMode', dest='visualizeMode',
                        help='visualization mode',
                        default='', type=str)    
    parser.add_argument('--suffix', dest='suffix',
                        help='suffix to distinguish experiments',
                        default='normal_augment', type=str)    
    parser.add_argument('--useCache', dest='useCache',
                        help='use cache instead of re-computing existing examples',
                        default=0, type=int)    
    parser.add_argument('--dataFolder', dest='dataFolder',
                        help='data folder',
                        default='/gruvi/Data/chenliu/ScanNet/scans/', type=str)
    parser.add_argument('--labelFile', dest='labelFile',
                        help='path to scannetv2-labels.combined.tsv',
                        default='/gruvi/Data/chenliu/ScanNet/tasks/scannetv2-labels.combined.tsv', type=str)
    parser.add_argument('--split', dest='split',
                        help='data split: [train, test, val]',
                        default='val', type=str)    
    
    args = parser.parse_args()
    return args
