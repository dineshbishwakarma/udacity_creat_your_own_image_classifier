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

parser.add_argument('--json_file', type=str, default='cat_to_name.json', help='Allows user to enter custom JSON file for category names.')
parser.add_argument('--test_file', type=str, default='flowers/train/43/image_02364.jpg', help='Allows user to run prediction on a given image.')
parser.add_argument('--checkpoint_file', type=str, default='vgg16_checkpoint.pth', help='Allows user to input a checkpoint file to load/build model from.')
parser.add_argument('--topk', type=int, default=5, help='Allows user to enter the top "k" predictions suggested by the model.')
parser.add_argument('--device', type=str, default='gpu', help='Enter the device to train the model i.e."gpu" or "cpu"')


#Maps parser arguments to variables for ease of use later
cl_inputs = parser.parse_args()

json_file = cl_inputs.json_file
test_file = cl_inputs.test_file
checkpoint_file = cl_inputs.checkpoint_file
topk = cl_inputs.topk
gpu = cl_inputs.device

#Imports inputted JSON file
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)
     
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    checkpoint = torch.load(filepath, map_location=map_location)
    
    arch = checkpoint['arch']
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    
    
    model.classifier = checkpoint['classifier']
    model.class_to_index = checkpoint['class_to_idx']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

model = load_checkpoint(filepath=checkpoint_file)
print(model)

# Image Preprocessing
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # DONE: Process a PIL image for use in a PyTorch model
    picture = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    tensor = transform(picture)
    
    return tensor

test_image = test_file
processed_image = process_image(test_image)
print(processed_image.shape)


def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu == 'gpu':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif gpu == 'cpu':
        device ='cpu'
    # DONE: Implement the code to predict the class from an image file
    #Processing image
    image = process_image(image_path)
    image = image.float().unsqueeze_(0)
    imgs = image.to(device)
    
    
    model.to(device)
    model.eval()   
          
    with torch.no_grad():
        out = model.forward(imgs)
        ps = torch.exp(out)
        
        pbs, inds = torch.topk(ps,topk)
        pbs = [float(pb) for pb in pbs[0]]
        inv_map = {val:key for key, val in model.class_to_index.items()}
        clss = [inv_map[int(idx)] for idx in inds[0]]
    return pbs, clss

test_image = test_file
probabilities, classes = predict(image_path=test_image,model=model,topk=topk, gpu=gpu)
flower_names = [cat_to_name[i] for i in classes]
print("probabilities:\n", probabilities)
print("classes:\n",flower_names)