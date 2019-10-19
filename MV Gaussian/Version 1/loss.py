# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:45:17 2019

@author: Admin
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.stats import multivariate_normal as mvn
import config


# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not, features1, features2, num_features, flag):
    #inputs: y = logit; t = labels (cuda); ind = index; noise_or_not = boolean array; 
    #inputs: features = data type?; flag = indicates when to start using the model
    
    # statistics for Gaussian models
    global N1
    global N2
    global mean1
    global mean2
    global cov1
    global cov2
    
    num_classes = 10 # For MNIST and CIFAR10
    labels = t.cpu().numpy()
    
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.cpu().detach().numpy())#.cuda()
    loss_1_sorted = loss_1[ind_1_sorted]
    
    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.cpu().detach().numpy())#.cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    if flag == 1:
        remember_rate = 1 - forget_rate
    else:
        remember_rate = 1 - forget_rate
        
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    # low-loss instances to keep
    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    num = ind_1_update.size
    
    # when epoch > args.use_model, flag == 1
    if flag == 1:
        # use cdf of Gaussian model to determine which instances to use for updating weights
        comp1 = np.zeros(num_remember,)
        comp2 = np.zeros(num_remember,)
        check1 = -1*np.ones(num,)
        check2 = -1*np.ones(num,)
        
        for j in range(num_classes):
            class1 = mvn(mean=config.mean1[:,j]/config.N1[0,j], cov=np.diag(config.cov1[:,j]/(config.N1[0,j]-1)), allow_singular = True)
            pdf = class1.pdf(features1[ind_1_update].detach().numpy().reshape(num, num_features))
            #cdf = class1.cdf(features1[ind_1_update].detach().numpy().reshape(num_remember, num_features))
            for i in range(num_remember):
                if comp1[i] < pdf[i]:
                    check1[i] = j # the class with the larget pdf will be the assigned label
                    comp1[i] = pdf[i] # used to keep track of the max. pdf
                 
            class2 = mvn(mean=config.mean2[:,j]/config.N2[0,j], cov=np.diag(config.cov2[:,j]/(config.N2[0,j]-1)), allow_singular = True)
            pdf = class2.pdf(features2[ind_2_update].detach().numpy().reshape(num, num_features))
            #cdf = class2.cdf(features1[ind_1_update].detach().numpy().reshape(num_remember, num_features))
            for i in range(num_remember):
                if comp2[i] < pdf[i]:
                    check2[i] = j # the class with the larget pdf will be the assigned label
                    comp2[i] = pdf[i] # used to keep track of the max. pdf
                    
        # order the indexes to see if they match 
        #ind_1_sorted = np.argsort(ind_1_update)#.cuda()
        #ind_2_sorted = np.argsort(ind_2_update)#.cuda()
        
        bool_array = labels[ind_1_update] == check1 #or ind_1_sorted == ind_2_sorted
        ind_1_update = ind_1_update[bool_array]
        bool_array = labels[ind_2_update] == check2
        ind_2_update = ind_2_update[bool_array]
    
    
    """ Trying to prevent the Gaussian model reducing the number of confident samples below some threshold
    if ind_1_update.size < 0.6*num:
        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]
    """
            
    # use low-loss/confident (for later epochs) instances to build Gaussian model- N (number of instances) covariance and mean
    for j in range(num_classes): 
        #mean1
        i_array = features1[ind_1_update].detach().numpy()[labels[ind_1_update]==j] 
        i_array = i_array.reshape(i_array.shape[0], num_features) #n x features
        config.mean1[:,j] = config.mean1[:,j] + i_array.sum(axis=0).reshape(num_features) # sum columns (axis = 1 to sum rows)
        
        #cov1 = calculate (xi-mu_x)(yi-mu_y) and keep only diagonal (variance)
        reps = i_array.shape[0]
        config.N1[0,j] = config.N1[0,j]+reps
        meanQ = np.repeat(config.mean1[:,j]/config.N1[0,j], reps, axis=0).reshape(num_features, reps) #features x n
        Q = (np.transpose(i_array)-meanQ)
        Q = np.matmul(Q, np.transpose(Q))
        config.cov1[:,j] = config.cov1[:,j] + np.diag(Q) #get diagonal elements and put in cov
        
        #mean2 = calculte sum of features
        i_array = features2[ind_2_update].detach().numpy()[labels[ind_2_update]==j] 
        i_array = i_array.reshape(i_array.shape[0], num_features) #n x features
        config.mean2[:,j] = config.mean2[:,j] + i_array.sum(axis=0).reshape(num_features) # sum columns (axis = 1 to sum rows)
        
        #cov2 = calculate (xi-mu_x)(yi-mu_y) and keep only diagonal (variance)
        reps = i_array.shape[0]
        config.N2[0,j] = config.N2[0,j]+reps
        meanQ = np.repeat(config.mean2[:,j]/config.N2[0,j], reps, axis=0).reshape(num_features, reps) #features x n
        Q = (np.transpose(i_array)-meanQ)
        Q = np.matmul(Q, np.transpose(Q))
        config.cov2[:,j] = config.cov2[:,j] + np.diag(Q) #get diagonal elements and put in cov
    
    
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2

#torch.cuda.empty_cache()
    
