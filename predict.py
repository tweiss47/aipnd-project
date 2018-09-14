# predict.py
#
# Command line tool to predict the class of an image using a pretrained
# pytorch network
#
# Author: Taylor Weiss
# Class: Udacity - AI Programming with Python Nanodegree Program
# Project: Image Classifier


import argparse
import data
import network
import json
import os.path


# Define command line arguments
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

# validate and convert the image file
image_filename = args.input
if not os.path.isfile(image_filename):
    print('Unable to find image file', image_filename)
    exit()

# BUGBUG: useful for testing, but probably don't want to assume the
# image is from the image dataset
image_category = image_filename.split(os.sep)[-2]

image_data = data.process_image(args.input)

# create the network
device = 'cuda' if args.gpu else 'cpu'
model = network.load_network(args.checkpoint, device)

# predict
probs, classes = network.predict(image_data, model, device, topk=args.top_k)

# Load the category to name mapping if provided
cat_to_name = None
if args.category_names and os.path.isfile(args.category_names):
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

# output results
print('Image category:', image_category)
if cat_to_name:
    print('Image name:', cat_to_name[image_category])
print('Probabilities:', probs)
print('Classes:', classes)
if cat_to_name:
    names = [cat_to_name[cat] for cat in classes]
    print('Names:', names)
