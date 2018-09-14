# data.py
#
# Helper module for the Image Classifier project.
# Functions to assist in loading and processing the image data.
#
# Author: Taylor Weiss
# Class: Udacity - AI Programming with Python Nanodegree Program
# Project: Image Classifier


import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image


def create_data_loaders(data_dir):
    '''
    Initialize the data loaders. Assume that the data_dir has test, train
    and valid subdirectories containing torchvision ImageFolder compatible
    data sets.

    return a map containing each data loader keyed by the directory name
    for the data set, d['train'] for example.
    '''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False)

    return {'train': train_loader, 'test': test_loader, 'valid': valid_loader}


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    size_orig = im.size

    # resize torchvision.transforms.resize(256) reduces the shortest dimension to 256
    if size_orig[0] < size_orig[1]:
        size_reduced = (256, 256 * size_orig[1] / size_orig[0])
    else:
        size_reduced = (size_orig[0] / size_orig[1] * 256, 256)
    im.thumbnail(size_reduced)

    # center crop to 224x224
    width, height = im.size
    left = (width - 224) / 2
    upper = (height - 224) / 2
    im = im.crop((left, upper, left + 224, upper + 224))

    # get array data and normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = np.array(im)

    np_image = np_image / 255.0
    np_image = np_image - mean
    np_image = np_image / std

    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image)



