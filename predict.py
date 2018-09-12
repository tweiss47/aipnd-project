import argparse

parser = argparse.ArgumentParser(
    description='use a checkpoint to predict the name of an image'
)

parser.add_argument(
    'input',
    action='store',
    help='path to the image file'
)
parser.add_argument(
    'checkpoint',
    action='store',
    help='path to the checkpoint file'
)
parser.add_argument(
    '--top_k',
    action='store',
    help='number of likely classes to return',
    type=int,
    default=5
)
parser.add_argument(
    '--category_names',
    action='store',
    help='path to a json mapping of category labels to names'
)
parser.add_argument(
    '--gpu',
    action='store_true',
    help='use the gpu for prediction',
    default=False
)

args = parser.parse_args()
print(args)
