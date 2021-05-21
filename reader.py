from __future__ import print_function
import os
import shutil
import argparse
import shutil
import time
import numpy as np
from numpy import linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import *
import pickle


model = utils.__dict__['vgg16_bn']()
model.load_state_dict(torch.load('./cifar10_pruned.pth.tar'), strict=True)

print("model:\n", model.state_dict())

#print(model['features.0'])

"""
with open('mask.pickle', 'rb') as fr:
    mask = pickle.load(fr)

print(mask['features.0.weight'])


pat_set = np.array([[1,1,1,1,0,0,0,0,0],
                    [0,0,0,0,0,1,1,1,1],
                    [1,1,0,0,0,0,0,1,1]])

x = np.array([[[0.1,0.2,0.3,0.4,0.5,-0.6,-0.7,-0.8,-0.9],[1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1]]])
z = top_4_pat(x, pat_set)
y = top_k_kernel(x, 2)
u = x - z
v = x - y

X = torch.tensor([[[0.1,0.2,0.3,0.4,0.5,-0.6,-0.7,-0.8,-0.9],[1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1]]],
                 dtype = torch.float)


print('x')
print(x)

print('z')
print(z)

print('y')
print(y)

print('x norm:', la.norm(x))
print('X norm:', torch.norm(X))
"""

