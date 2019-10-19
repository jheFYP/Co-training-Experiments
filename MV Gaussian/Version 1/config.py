# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:51:12 2019

@author: Admin
"""
import numpy as np

num_classes = 10
num_features = 128

# initialise Gaussian models
N1 = np.zeros((1,num_classes))
cov1 = np.zeros((num_features, num_classes))
mean1 = np.zeros((num_features, num_classes))
N2 = np.zeros((1,num_classes))
cov2 = np.zeros((num_features, num_classes))
mean2 = np.zeros((num_features, num_classes))