import os
import os.path
import random
import cv2
import math
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as torch_utils_data

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS
NPLEN=7
NUM_CLASSES=1+len(CHARS)*NPLEN

class anprmodel(nn.Module):
    def __init__(self):
        super(anprmodel,self).__init__()
        self.num_classes=NUM_CLASSES
        self.conv1=nn.Conv2d(1,48,kernel_size=3,padding=1)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(48,64,kernel_size=3,padding=1)
        self.pool2=nn.MaxPool2d(kernel_size=(2,1),stride=(2,1))    
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))        
        self.fc1=nn.Linear(32*8*128,2048)
        self.fc2=nn.Linear(2048,self.num_classes)
        
    def forward(self,x): 
        x=F.relu(self.pool1(self.conv1(x)))
        x=F.relu(self.pool2(self.conv2(x)))
        x=F.relu(self.pool3(self.conv3(x)))
        x=x.view(-1,32*8*128)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
ifdef __name__ is '__main__':
    net=anprmodel()
    exit()
