# -*- coding: utf-8 -*-
"""
Last editied on Tue Sep 24 22:52:28 2019

@author: Jen
"""

import numpy as np
import torch
import torchvision
from torchvision import transforms
import math
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis as mahal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#load data
input_channel=1
num_classes=10
batch_size = 128
n_epoch = 200
train_dataset = torchvision.datasets.MNIST(root='./data/',
                            download=True,  
                            train=True, 
                            transform=transforms.ToTensor()
                     )

test_dataset = torchvision.datasets.MNIST(root='./data/',
                           download=True,  
                           train=False, 
                           transform=transforms.ToTensor()
                    )

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               drop_last=True,
                                               shuffle=True)
    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              drop_last=True,
                                              shuffle=False)

# initialize
imsize = 28
num_epoch = 5
N = np.zeros((1,num_classes))
cov = np.zeros((500, num_classes))
mean = np.zeros((500, num_classes))

#output after cnn reduces features
device = torch.device("cpu")
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        #out2 = x
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        out2=x # extract features to use for modelling
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), out2

    
def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, features = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
        labels = target.numpy()
        check = np.ones(128,)
        check2 = np.ones(128,)
        comp = np.ones(128,)
        dist = np.ones((1, num_classes))
        
        #0 to 9 classes
        for j in range(num_classes): 
            #mean
            i_array = features.detach().numpy()[labels==j] 
            i_array = i_array.reshape(i_array.shape[0], 500) #nx784
            mean[:,j] = mean[:,j] + i_array.sum(axis=0).reshape(500) # sum columns (axis = 1 to sum rows)
            
            #cov
            reps = i_array.shape[0]
            N[0,j] = N[0,j]+reps
            meanQ = np.repeat(mean[:,j]/N[0,j], reps, axis=0).reshape(500, reps) #784xn
            Q = (np.transpose(i_array)-meanQ)
            Q = np.matmul(Q, np.transpose(Q))
            cov[:,j] = cov[:,j] + np.diag(Q) #get diagonal elements and put in cov
            
            #Gaussian
            if batch_idx > 80:
                # construct pdf for MV Gaussian (do not use cdf for computational time reasons)
                class1 = multivariate_normal(mean=mean[:,j]/N[0,j], cov=np.diag(cov[:,j]/(N[0,j]-1)), allow_singular = True)
                pdf = class1.pdf(features.detach().numpy().reshape(batch_size, 500))
                #print(pdf) #uncomment to see pdf output
                for i in range(128):
                    if pdf[i] > 0 and comp[i] < pdf[i]: # assume the largest pdf value (integrated give cdf) corresponds with highest probability
                        check[i] = j
                        comp[i] = pdf[i]
        #mahalanobis                
        if batch_idx > 80:
            x = features.detach().numpy()
            for k in range(batch_size):
                for j in range(num_classes):
                    dist[0,j] = mahal(x[k], mean[:,j]/N[0,j], np.diag(cov[:,j]/(N[0,j]-1)))
                    check2[k] = dist.argmin() # min value gives shortest distance
 
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
log_interval = 50
               
for epoch in range(n_epoch):
    train(log_interval, model, device, train_loader, optimizer, epoch)