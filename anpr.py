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
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

class vgg16(nn.Module):
    #def __init__(self,features,num_classes=1000):
    def __init__(self,num_classes=1000):
        super(vgg16,self).__init__()
        self.features=nn.Sequential(
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
        self.classifier=nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
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

class NPSET(data.Dataset):
    picroot='np'

    def __getitem__(self,index):
        label,img=self.labels[index], self.dataset[index]
        if self.data_transform is not None:
            img=self.data_transform(img)
        if self.label_transform is not None:
            label=self.label_transform(label)
        return label,img

    def __len__(self):
        return self.len

    def __init__(self,root,data_transform=None,label_transform=None):
        self.picroot=root
        self.data_transform=data_transform
        self.label_transform=label_transform

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
                m=filename.split('_')  #filename stype: xxxxxxxx_xxxxxxx_x.png
                labels.append(m[1])
                i=i+1
            self.dataset=imgs
            self.labels=labels
            self.len=len(filenames)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    model=vgg16()  #number class =?
    #model.features=torch.nn.DataParallel(model.features)
    #model.cuda()
    cudnn.benchmark=True
    batch_size=10
    transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                             ])
    npset = NPSET(root='np', data_transform=transform)
    nploader = torch.utils.data.DataLoader(npset, batch_size=batch_size, shuffle=False, num_workers=1)  #train
    npvalset=NPSET(root='npval', data_transform=transform)
    npvalloader=torch.utils.data.DataLoader(npvalset, batch_size=batch_size, shuffle=False, num_workers=1) #validate
    #criterion=nn.CrossEntropyLoss().cuda()
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters,0.1,momentum=0.9)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
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
            loss=criterion(output,target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            if i% 10 == 0:
                print('Eoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(i,len(nploader),loss=losses,top1=top1,top5=top5)

        #validate
        model.eval()
        for i, data in enumerate(npvalloader):
            (inputs, target)=data
            #target = target.cuda()
            input_var = torch.autograd.Variable(inputs, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            #
            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(npvalloader), loss=losses,
                       top1=top1, top5=top5))
            prec1=top1.avg
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'arch:vgg16 state_dict': model.state_dict(),
                'best_prec1': best_prec1
            })

