import argparse

parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--featmode', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--abvideonum', type=int,default=-1, help='num of abnormal videos (default: 2048)')
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--lr', type=str, default='[0.00001]*15000', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=24, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--dataset', default='sh', help='dataset to train on (default: )')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=600, help='maximum iteration to train (default: 100)')
