# -*- coding: utf-8 -*-
"""
Last edited on Wed Sep 25 22:53:24 2019

@author: Jen
"""

import numpy as np
import torch
import torchvision
from torchvision import transforms
from sklearn.neighbors import KernelDensity
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Set parameters
input_channel=1 #MNIST has 1 channel; CIFAR10 has 3
num_classes=10
batch_size = 128
n_epoch = 5
imsize = 28 # 32 for CIFAR10

#load data
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
        out2=x
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), out2

# initialize arrays
class0 = np.array([1])
class1 = np.array([1])
class2 = np.array([1])
class3 = np.array([1])
class4 = np.array([1])
class5 = np.array([1])
class6 = np.array([1])
class7 = np.array([1])
class8 = np.array([1])
class9 = np.array([1])

""" 
# sensitive to bandwidth parameter (https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html):
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut

bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut(len(x)))
grid.fit(x[:, None]);
"""

def train(log_interval, model, device, train_loader, optimizer, epoch):
    global class0
    global class1
    global class2
    global class3
    global class4
    global class5
    global class6
    global class7
    global class8
    global class9
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, features = model(data)
        features = features.detach().numpy()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            

        
        # Add samples to each class array for KDE
        if class0.shape == [1]:
            class0 = np.concatenate((class0, features[target.numpy() == 0]))
            class1 = np.concatenate((class0, features[target.numpy() == 1]))
            class2 = np.concatenate((class0, features[target.numpy() == 2]))
            class3 = np.concatenate((class0, features[target.numpy() == 3]))
            class4 = np.concatenate((class0, features[target.numpy() == 4]))
            class5 = np.concatenate((class0, features[target.numpy() == 5]))
            class6 = np.concatenate((class0, features[target.numpy() == 6]))
            class7 = np.concatenate((class0, features[target.numpy() == 7]))
            class8 = np.concatenate((class0, features[target.numpy() == 8]))
            class9 = np.concatenate((class0, features[target.numpy() == 9]))
        else:
            class0 = features[target.numpy()==0]
            class1 = features[target.numpy()==1]
            class2 = features[target.numpy()==2]
            class3 = features[target.numpy()==3]
            class4 = features[target.numpy()==4]
            class5 = features[target.numpy()==5]
            class6 = features[target.numpy()==6]
            class7 = features[target.numpy()==7]
            class8 = features[target.numpy()==8]
            class9 = features[target.numpy()==9]

        # Gaussian KDE    
        # fit(X), score_samples: X = array (num_samples, num_features)    
        bandwidth = 0.25
        kde0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(class0)
        kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(class1)
        kde2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(class2)
        kde3 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(class3)
        kde4 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(class4)
        kde5 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(class5)
        kde6 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(class6)
        kde7 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(class7)
        kde8 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(class8)
        kde9 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(class9)
        
        if epoch > 3:
            check = np.zeros(128,)
            for i in range(128):
                # min score will be the most probable class    
                feat = features[i].reshape(1,-1)
                val = np.array([(kde0.score_samples(feat), kde1.score_samples(feat), kde2.score_samples(feat), kde3.score_samples(feat), kde4.score_samples(feat), kde5.score_samples(feat), kde6.score_samples(feat), kde7.score_samples(feat), kde8.score_samples(feat), kde9.score_samples(feat))])
                check[i] = val.argmax()
                
            boolar = check == target.numpy()
            print(np.sum(boolar)/128)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
log_interval = 50
               
for epoch in range(n_epoch):
    train(log_interval, model, device, train_loader, optimizer, epoch)
       
        
