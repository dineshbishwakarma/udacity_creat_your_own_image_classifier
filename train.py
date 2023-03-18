# Imports here
import torch
from torchvision import datasets,transforms, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
import json
from workspace_utils import active_session
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse 

#Brings in arguments from CLI
parser = argparse.ArgumentParser()

#UPDATE 4: Added data_dir and save_dir as input to the ArgumentParser
parser.add_argument('--data_dir', type=str, default="./flowers/", help='Determines which directory to pull information from')
parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', help='Enables user to choose directory for saving')
parser.add_argument('--arch', type=str, default='vgg16', help='Determines which architecture you choose: "densenet121" or "vgg16".')
parser.add_argument('--hidden_units', type=int, default=512, help='Dictates the hidden units for the hidden layer')
parser.add_argument('--learning_rate', type=float, default=.001, help='Dictates the rate at which the model does its learning')
parser.add_argument('--epochs', type=int, default=10, help='Determines number of cycles to train the model')
parser.add_argument('--device', type=str, default='gpu', help='Enter the device to train the model i.e."gpu" or "cpu"')



#Maps parser arguments to variables for ease of use later
cl_inputs = parser.parse_args()

# Location of data
data_dir = cl_inputs.data_dir

# Location to save model
save_dir = cl_inputs.save_dir

# Network Architecture
arch = cl_inputs.arch

# Hyperparameters 
lr = cl_inputs.learning_rate
hidden_units = cl_inputs.hidden_units
epochs = cl_inputs.epochs

# Device to train
gpu = cl_inputs.device

# Choose device to train
if gpu == 'gpu':    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Model is training on:',device) 
    
elif gpu == 'cpu':
    device = torch.device("cpu")
    print('Model is training on:',device)
    
# Extract the data
#data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


# DONE: Load the datasets with ImageFolder

# Training Datasets
trainset = datasets.ImageFolder(train_dir,transform=data_transforms)

# Testing Datasets
testset = datasets.ImageFolder(test_dir,transform=data_transforms)

# Validation Datasets
validset = datasets.ImageFolder(valid_dir,transform=data_transforms)

# DONE: Using the image datasets and the trainforms, define the dataloaders

# Training Set Loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Test Set Loader 
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

# Valiation Set Loader 
validloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=True)

# Label Mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    classes = len(cat_to_name)
    
# Buildign and Training the Classifier

# Defining the Network
def Network(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
        
    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(in_features, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('drop',nn.Dropout(p=0.05)),
                              ('fc2',nn.Linear(hidden_units,classes)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    
    return model

model = Network(arch=arch, hidden_units=hidden_units)

# Optimizer and Loss Fucntion 
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

model.to(device)

# Function to Train and Validate Dataset

def train(model, trainloader, validloader, criterion, optimizer, gpu):
    """
    Parameters: 
    ---------------------------------------
    model : Model to rain your network
    trainloader : training dataset
    valiloader : vlaidation dataset
    criterion: Loss Function
    optimizer : Optimimization Alogrithms
    gpu : device
    -------------------------------------------
    Returns: 
    -------------------------------------------
    None
    """
    
    #epochs = epochs
    steps = 0 # Steps here is batches. 1 batch = 64 images
    running_loss = 0
    print_every = 40 # the model trains on 40 batches of images at a time
    print('Training Started!')
    with active_session():
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to device
                inputs, labels = inputs.to(gpu), labels.to(gpu)

                logps = model.forward(inputs)
                loss = criterion(logps, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(gpu), labels.to(gpu)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()
    print('Done With the Training!')
    
# Train the Network
train(model=model,trainloader=trainloader,validloader=validloader,criterion=criterion, 
     optimizer=optimizer,gpu=device)

print('Starting Model Validation on Test Set!')

# DONE: Do validation on the test set
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test accuracy: {(accuracy/len(testloader)) * 100:.1f}%")

print('Saving checkpoint...!')

# DONE: Save the checkpoint
model.class_to_idx = trainset.class_to_idx
checkpoint = {'arch' :arch,
              'gpu':gpu,
              'input_size': 25088 if arch == 'vgg16' else 1024,
              'output_size': classes,
              'hidden_layer': hidden_units,
              'classifier':model.classifier,
              'state_dict':model.state_dict(),
              'optimnizer_state_dict': optimizer.state_dict,
              'class_to_idx': model.class_to_idx}
torch.save(checkpoint, f'{arch}_checkpoint.pth')