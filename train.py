import argparse

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

args = parser.parse_args()
print(args)
print(args.arch)


# https://docs.python.org/3/library/functions.html#getattr
# can use getattr to bind a method (or attribute) by name
