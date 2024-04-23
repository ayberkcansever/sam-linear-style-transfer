import torch
import numpy as np
from imageio import imread, imwrite
from skimage.transform import resize
import os
import json

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def loadJSON(file):
    with open(file) as f:
        result = json.load(f)
    return result

def saveJSON(data,file):
    with open(file, 'w') as outfile:
        json.dump(data, outfile)

def loadImage(filename, asTensor = True, imagenet_mean = False, shape = None):

    image = imread(filename)

    if image.ndim == 2:
        image = np.stack((image,image,image),axis=2)

    image = image[:,:,:3]/255 # No alpha channels

    if shape is not None:
        image = resize(image, shape)

    if imagenet_mean:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean)/std

    # Reorganize for torch
    if asTensor:
        image = np.transpose(image,(2,0,1))

        image = np.expand_dims(image,axis=0)

        return torch.tensor(image,dtype=torch.float)
    else:
        return image
    
def loadMask(filename, asTensor = True, shape = None):

    image = imread(filename)
    
    if image.ndim == 3:
        image = image[:,:,0]

    if shape is not None:
        image = resize(image, shape) 

    mask = image >= 128
        
    # Reorganize for torch
    if asTensor:
        mask = np.expand_dims(mask,axis=0)

        return torch.tensor(mask,dtype=torch.bool)
    else:
        return mask
    
def toTensor(numpy_data):
    image = np.transpose(numpy_data,(2,0,1))
    image = np.expand_dims(image,axis=0)
    return torch.tensor(image,dtype=torch.float)

def toTorch(numpy_data):
    return toTensor(numpy_data)

def toNumpy(tensor, permute=True):
    image = np.squeeze(tensor.detach().clone().cpu().numpy())
    if permute:
        image = np.transpose(image,(1,2,0))
    return image

def saveImage(filename,image):
    image = image.clamp(0,1)
    image = image.detach().clone().cpu().numpy()[0].transpose(1,2,0)
    image = (255*image).astype(np.uint8)
    imwrite(filename,image)
