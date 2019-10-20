# -*- coding: utf-8 -*-
"""
@inproceedings{han2018coteaching,
  title={Co-teaching: Robust training of deep neural networks with extremely noisy labels},
  author={Han, Bo and Yao, Quanming and Yu, Xingrui and Niu, Gang and Xu, Miao and Hu, Weihua and Tsang, Ivor and Sugiyama, Masashi},
  booktitle={NeurIPS},
  pages={8535--8545},
  year={2018}
}
Modified by JHe
"""

import os
import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10
from data.mnist import MNIST
from model import CNN
import argparse, sys
import numpy as np
import datetime
import shutil
import time
import config

from loss import loss_coteaching

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = './results')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)
parser.add_argument('--forget_rate', type = float, help = 'forget rate, default = None', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=124)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=0, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400) #400
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--num_features', type=int, default=128)
parser.add_argument('--use_model', type=int, default = 20)

args = parser.parse_args()

torch.cuda.empty_cache()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Hyper Parameters
batch_size = 128 
learning_rate = args.lr 

# load dataset
if args.dataset=='mnist':
    input_channel=1
    imsize = 28
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = MNIST(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                         )
    
     
    
    test_dataset = MNIST(root='./data/',
                               download=True,  
                               train=False, 
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                        )
    
if args.dataset=='cifar10':
    input_channel=3
    imsize = 32
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                           )
    
    test_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                          )

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

# Boolean array indicating if data label is noisy or not
noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)
   
save_dir = args.result_dir +'/' +args.dataset+'/coteaching'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
#    os.system('mkdir -p %s' % save_dir)
    
#if not os.path.isdir(args.tensorboard_dir):
#    os.mkdir(args.tensorboard_dir)

model_str=args.dataset+'_coteaching_'+args.noise_type+'_'+str(args.noise_rate)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target = target.type(torch.LongTensor).cuda()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2, flag):
    print('\nTraining %s...' % model_str)
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 
    
    global N1
    global N2
    global mean1
    global mean2
    global cov1
    global cov2

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu()
        ind = ind.numpy().transpose()
        if i>args.num_iter_per_epoch:
            break
      
        t = labels.numpy()
        images = Variable(images).cuda()  #torch.autograd = Variable
        labels = Variable(labels).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        
        # Forward + Backward + Optimize
        logits1, features1 = model1(images)
        features1 = features1.cpu() # dimensions: batch_size x num_features
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        logits2, features2 = model2(images)
        features2 = features2.cpu()
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2+=1
        train_correct2+=prec2
        
        # initilise Gaussian distribution
        if epoch == 1:
            for j in range(num_classes): 
                #mean1
                i_array = features1.detach().numpy()[t==j] 
                i_array = i_array.reshape(i_array.shape[0], args.num_features) #n x features
                config.mean1[:,j] = config.mean1[:,j] + i_array.sum(axis=0).reshape(args.num_features) # sum columns (axis = 1 to sum rows)
                
                #cov1
                reps = i_array.shape[0]
                config.N1[0,j] = config.N1[0,j]+reps
                meanQ = np.repeat(config.mean1[:,j]/config.N1[0,j], reps, axis=0).reshape(args.num_features, reps) #features x n
                Q = (np.transpose(i_array)-meanQ)
                Q = np.matmul(Q, np.transpose(Q))
                config.cov1[:,j] = config.cov1[:,j] + np.diag(Q) #get diagonal elements and put in cov
                
                #mean2
                i_array = features2.detach().numpy()[t==j] 
                i_array = i_array.reshape(i_array.shape[0], args.num_features) #n x features
                config.mean2[:,j] = config.mean2[:,j] + i_array.sum(axis=0).reshape(args.num_features) # sum columns (axis = 1 to sum rows)
                
                #cov2
                reps = i_array.shape[0]
                config.N2[0,j] = config.N2[0,j]+reps
                meanQ = np.repeat(config.mean2[:,j]/config.N2[0,j], reps, axis=0).reshape(args.num_features, reps) #features x n
                Q = (np.transpose(i_array)-meanQ)
                Q = np.matmul(Q, np.transpose(Q))
                config.cov2[:,j] = config.cov2[:,j] + np.diag(Q) #get diagonal elements and put in cov
        
        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not, features1, features2, args.num_features, flag)
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.cpu().item(), loss_2.cpu().item(), np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list
            




# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1, _ = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2, _ = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
        
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')
    cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn1.cuda()
    print(cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    
    cnn2 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn2.cuda()
    print(cnn2.parameters)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    mean_pure_ratio1=0
    mean_pure_ratio2=0

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n')

    epoch=0
    train_acc1=0
    train_acc2=0
    global flag
    flag = 0 #flag is a parameter of loss_coteaching and indicates when to start using the statistical model
    
    global N1
    global N2
    global mean1
    global mean2
    global cov1
    global cov2

    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + "\n")
    
    # training
    for epoch in range(1, args.n_epoch):
        if epoch % 50 == 0 and epoch > 0:
            time.sleep(300) #let GPU cool for 5mins
            
        if epoch >= args.use_model:
            flag = 1
            
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2, flag)
        # evaluate models
        test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
        #clear cuda
        torch.cuda.empty_cache()
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")

if __name__=='__main__':
    main()
