# network.py
#
# Helper module for the Image Classifier project.
# Functions to assist in building, training and saving the image classifier
# network.
#
# Author: Taylor Weiss
# Class: Udacity - AI Programming with Python Nanodegree Program
# Project: Image Classifier


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


#######################################################################
# Public Interface
#######################################################################


def load_network(checkpoint, device):
    '''
    Create a network from a saved checkpoint
    '''
    # Load the model onto the correct device
    params = torch.load(checkpoint, map_location=device)

    # I have a vgg16 model that was trained before I started serializing the
    # model name. So if model_type wasn't specified default to vgg16
    if 'model_type' not in params:
        params['model_type'] = 'vgg16'

    return create_network_with(params)


def create_network(arch, hidden_layers, output_size = 102, drop_p = 0.5, lr = 0.001):
    '''
    Wrapper for create_network_with()
    '''
    params = {
        'model_type': arch,
        'hidden_layers': hidden_layers,
        'output_size': output_size,
        'drop_p': drop_p,
        'lr': lr
    }
    return create_network_with(params)


def train_network(model, train_loader, valid_loader, epochs, device, print_every=40, fake=False):
    '''
    Train our image classification model

    model - the image classification model
    train_loader - DataLoader for training data
    valid_loader = DataLoader for validation data
    epochs - number of epochs to train
    device - 'cuda' or 'cpu'
    print_every - number of steps to print out progress
    fake - test parameter, cpu training is SLOW, only do one training pass
    '''
    optimizer = model.optimizer
    criterion = nn.NLLLoss()

    # save class to index mapping for later prediction reporting
    model.class_to_idx = train_loader.dataset.class_to_idx

    print("Start model training")

    # Use cuda if available
    model.to(device)

    if fake:
        epochs = 1
        print_every = 1

    steps = 0
    for e in range(epochs):
        # ensure that we are in training mode to start
        model.train()

        # loss to report
        running_loss = 0

        # iterate through training data
        for inputs, labels in train_loader:
            steps += 1

            # move to device cuda if available
            inputs, labels = inputs.to(device), labels.to(device)

            # training steps
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # report loss and current accuracy
            running_loss += loss.item()
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                if fake:
                    test_loss, accuracy = 0.0, 0.0
                else:
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, valid_loader, criterion, device)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

                running_loss = 0
                # Make sure training is back on
                model.train()

            if fake:
                break

    print("End model training")


def save_checkpoint(model, filepath):
    '''
    Serialize the network to filepath
    '''
    checkpoint = dict(model.params)
    checkpoint['class_to_idx'] = model.class_to_idx
    checkpoint['state_dict'] = model.classifier.state_dict()
    checkpoint['opt_state_dict'] = model.optimizer.state_dict()
    torch.save(checkpoint, filepath)


def predict(image_data, model, device, topk = 5):
    '''
    Return the topk probabilities and labels for the image using the trained model

    probs - nparray of probabilities
    classes - list of the class labels
    device - 'cuda' or 'cpu'
    topk - number to return

    returns probs, classes
        probs - nparray of probabilities
        classes - list of the class labels
    '''

    # run the image data through the model
    model.to(device)
    model.eval()
    with torch.no_grad():
        # ensure model and data are in the same precision
        # package image data using unsqueeze
        output = model.double().forward(image_data.unsqueeze(0).to(device))
        ps = torch.exp(output)

    # extract probabilities and classes from the model output
    top = ps.topk(topk)
    probs = top[0][0].numpy()
    indices = top[1][0].numpy().tolist()

    # convert indices to labels
    idx_to_class = dict([(v, k) for k, v in model.params['class_to_idx'].items()])
    labels = [idx_to_class[idx] for idx in indices]

    return probs, labels


#######################################################################
# Internal
#######################################################################


def create_network_with(params):
    '''
    Create a new test network from a pretrained feature detection model and
    input parameters defining the state of the model. Can be used to create
    a new model if the params dict does not define state parameters. Otherwise
    the state parameters are loaded into the new classifier.

    params - dict containing create and state parameters:
        'model_type' the name of a pretrained model architecture in
                     torchvision.models
        'output_size' size of the output data set
        'hidden_layers' a list of hidden layer sizes
        'drop_p' training dropout probability
        'lr' learning reate
        'state_dict' the state dictionary for a saved model
        'opt_state_dict' the state dictionary for the optimizer

    Returns the model and optimizer.
    '''
    # create the feature detection model
    model_constructor = getattr(models, params['model_type'])
    model = model_constructor(pretrained=True)

    # freeze the parameters of the feature detection model
    for param in model.parameters():
        param.requires_grad = False

    # create the new classifier
    input_size = get_classifier_in_features(model)

    # Create an instance of the network to replace the classifier
    classifier = FeedForwardNetwork(input_size=input_size,
                                    output_size=params['output_size'],
                                    hidden_layers=params['hidden_layers'],
                                    drop_p=params['drop_p'])

    # Load saved state if provided
    if 'state_dict' in params:
        classifier.load_state_dict(params['state_dict'])

    # Replace the origianl classifier
    model.classifier = classifier

    # Create the optimizer and load state if necessary
    optimizer = optim.Adam(model.classifier.parameters(), lr=params['lr'])
    if 'opt_state_dict' in params:
        optimizer.load_state_dict(params['opt_state_dict'])

    model.optimizer = optimizer
    model.params = params
    return model


def get_classifier_in_features(model):
    '''Return the number of inputs to the classifier for the given model'''
    first_param = next(iter(model.classifier.parameters()))
    return first_param.size()[1]


def validation(model, testloader, criterion, device):
    '''
    Run testloader data through the model and return loss and accuracy

    Relies on caller to turn off gradient calculation
    Relies on caller to put model in eval mode
    '''
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


# Helper class from  Lesson 5 to create a classifier with different paramters
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network'''

        # Forward through each layer in hidden_layers, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)
