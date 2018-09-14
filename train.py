# train.py
#
# Command line tool to train an image classifier using a torchvision model
#
# Author: Taylor Weiss
# Class: Udacity - AI Programming with Python Nanodegree Program
# Project: Image Classifier


import argparse
import data
import network
import os.path
import time


# Handle command line parameters
parser = argparse.ArgumentParser(
    description='train a new pytorch network on a dataset'
)
parser.add_argument(
    'data_dir',
    action='store',
    help='directory containing data to train the model'
)
parser.add_argument(
    '--save_dir',
    action='store',
    help='directory to save checkpoint file',
    default='.'
)
parser.add_argument(
    '--arch',
    action='store',
    help='model architecture',
    default='vgg13'
)
parser.add_argument(
    '--hidden_units',
    action='append',
    nargs='+',
    type=int,
    help='hidden layers for classifier',
    default=[]
)
parser.add_argument(
    '--epochs',
    action='store',
    type=int,
    help='number of training epochs',
    default=5
)
parser.add_argument(
    '--gpu',
    action='store_true',
    help='use the gpu for training',
    default=False
)
# For Testing Only
parser.add_argument(
    '--fake',
    action='store_true',
    help='for testing only',
    default=False
)

args = parser.parse_args()

# validate the data directory
if not os.path.isdir(args.data_dir):
    print(args.data_dir, 'is not a valid data directory.')
    exit()

# create data loader
data = data.create_data_loaders(args.data_dir)

# create the network
hidden_units = [4000, 1000, 200]
if len(args.hidden_units) > 0:
    hidden_units = args.hidden_units[0]

model = network.create_network(args.arch, hidden_units)

# train the network
device = 'cuda' if args.gpu else 'cpu'
network.train_network(model, data['train'], data['valid'], args.epochs, device, fake=args.fake)

# save the checkpoint
checkpoint_filename = 'checkpoint_' + args.arch + '_' + time.strftime("%Y-%m-%dT%H-%M-%S")
checkpoint_path = args.save_dir + '/' + checkpoint_filename
network.save_checkpoint(model, checkpoint_path)
print("Network written to", checkpoint_path)

