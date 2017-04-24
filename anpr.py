__author__ = 'wang'

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
#PROVINCE="黑吉辽京津内冀鲁豫徽苏沪浙赣闽粤鄂湘云贵川渝藏青宁新陕甘宁晋" #30
CHARS = LETTERS + DIGITS
NPLEN=7
NUM_CLASSES=1+len(CHARS)*NPLEN

conv=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,padding=1), #layer1, inputs single channel,224*224
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,128,kernel_size=3,padding=1), #layer2 inputs 64 channel,112*112
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(128,256,kernel_size=3,padding=1), #layer3 inputs 128 channel,56*56
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(256,512,kernel_size=3,padding=1), #layer4 inputs 256 channel,28*28
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1), #layer5 inputs 512 channel,14*14
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
    )

class vgg16train(nn.Module):
    def __init__(self): #36*7+1=253   36*6+1=217
        super(vgg16train,self).__init__()
        self.features=conv
        self.classifier=nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes)
        )
        #initialize_weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
    
class vgg16detect(nn.Module):    
    def __init__(self):
        super(vgg16detect,self).__init__()
        self.features=conv
        self.classifier=nn.Sequential(
            nn.Conv2d(512,4096,kernel_size=7,padding=1),  #padding=1?
            nn.ReLU(inplace=True),
            nn.Conv2d(4096,2048,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096,num_classes,kernel_size=1,padding=1),
            #nn.ReLU(inplace=True),
        )
    def forward(self,x):  #是否需要
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x    

class anprmodel(nn.Module):
    def __init__(self):
        super(anprmodel,self).__init__()
        self.num_classes=NUM_CLASSES
        self.conv1=nn.Conv2d(1,48,kernel_size=5,padding=2)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(48,64,kernel_size=3,padding=1)
        self.pool2=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))    
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))        
        self.fc1=nn.Linear(28*28*128,2048)
        self.fc2=nn.Linear(2048,NUM_CLASSES)
        
    def forward(self,x): 
        x=F.relu(self.pool1(self.conv1(x)))  #224*224
        x=F.relu(self.pool2(self.conv2(x)))  #112*112
        x=F.relu(self.pool3(self.conv3(x)))  #56*56
        #x=x.view(-1,32*8*128)
        x=x.view(-1,28*28*128)                   #28*28
        x=F.relu(self.fc1(x))
        x=self.fc2(x)                                       #253
        return x

class NPSET(torch_utils_data.Dataset):
    picroot='np'
   
    def code_to_vec(self,p, code):
        def char_to_vec(c):
            y = np.zeros((len(CHARS),))
            y[CHARS.index(c)] = 1.0
            return y
        c = np.vstack([char_to_vec(c) for c in code])
        return np.concatenate([[1. if p else 0], c.flatten()])

    def __getitem__(self,index):
        label,img=self.labels[index], self.dataset[index]
        if self.data_transform is not None:
            img=self.data_transform(img)
        labelarray=self.code_to_vec(1,label)
        #if self.label_transform is not None:
        #    labelarray=self.label_transform(labelarray)
        return img,labelarray

    def __len__(self):
        return self.len

    def __init__(self,root,data_transform=None):
        self.picroot=root
        self.data_transform=data_transform

        if not os.path.exists(self.picroot):
            raise RuntimeError('{} doesnot exists'.format(self.picroot))
        for root,dnames,filenames in os.walk(self.picroot):
            imgs=np.ndarray(shape=(len(filenames),1,224,224),dtype=np.float)
            labels=[]
            i=0
            for filename in filenames:
                picfilename=os.path.join(self.picroot,filename)  #file name:
                im=cv2.imread(picfilename,cv2.IMREAD_GRAYSCALE)
                im=cv2.resize(im,(224,224))
                #(thresh, im) = cv2.threshold(im, 32, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                #im=cv2.erode(im,self.kernel)
                #im=cv2.dilate(im,self.kernel)
                #im=cv2.GaussianBlur(im,(5,5),0.1)
                #(thresh, im) = cv2.threshold(im, 32, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                imgs[i][0]=im/255
                m=filename.split('_')  #filename style: xxxxxxxx_xxxxxxx_x.png
                labels.append(m[1])
                i=i+1
            self.dataset=imgs
            self.labels=labels
            self.len=len(filenames)


if __name__ == '__main__':
    model=anprmodel()
    #model.features=torch.nn.DataParallel(model.features)
    #model.cuda()
    #cudnn.benchmark=True
    batch_size=4
    data_transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                             ])
    npset = NPSET(root='/home/wang/git/anpr/np', data_transform=data_transform)
    nploader = torch.utils.data.DataLoader(npset, batch_size=batch_size, shuffle=False, num_workers=1)  #train
    npvalset=NPSET(root='/home/wang/git/anpr/npval', data_transform=data_transform)
    npvalloader=torch.utils.data.DataLoader(npvalset, batch_size=batch_size, shuffle=False, num_workers=1) #validate
    criterion=nn.MultiLabelMarginLoss()
    optimizer=torch.optim.SGD(model.parameters(),0.1,momentum=0.9)

for epoch in range(0,1):
    #Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr=0.1*(0.1**(epoch//30))
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    #train
    model.train()
    for i,data in enumerate(nploader):
        inputs,target = data
        #target=target.cuda()
        input_var=torch.autograd.Variable(inputs)
        target_var=torch.autograd.Variable(target)
        output=model(input_var)
        #porcess loss
        o=torch.FloatTensor(np.reshape(output.data.numpy()[:,1:],(-1,len(CHARS))))
        t=torch.LongTensor(np.array(np.reshape(target_var.data.numpy()[:,1:],(-1,len(CHARS))),np.long))
        chararcter_loss=cerition(torch.autograd.Variable(o), torch.autograd.Variable(t))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        if i% 12 == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}'.format(
                  epoch, i * len(data), len(nploader.dataset),
                  100. * i / len(nploader), loss.data[0]))
        
    #validate
    model.eval()
    for i, data in enumerate(npvalloader):
        (inputs, target)=data
        #target = target.cuda()
        input_var = torch.autograd.Variable(inputs, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        output = model(input_var)
        #porcess loss
        o=torch.FloatTensor(np.reshape(output.data.numpy()[:,1:],(-1,len(CHARS))))
        t=torch.LongTensor(np.array(np.reshape(target_var.data.numpy()[:,1:],(-1,len(CHARS))),np.long))
        chararcter_loss=cerition(torch.autograd.Variable(o), torch.autograd.Variable(t))
        #
        bo=torch.FloatTensor(output.data.numpy()[:,1:])
        bt=
        b=0,d=0
        for k < o.size(0)
        #
        if i% 12 == 0:
             print('Test Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}'.format(
                  epoch, i * len(data), len(nploader.dataset),
                  100. * i / len(nploader), loss.data[0]))
        prec1=top1.avg
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
    if is_best:
        torch.save({
            'epoch': epoch + 1,
            'arch':vgg16,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        })
