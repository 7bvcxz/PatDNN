from __future__ import print_function
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import *
import pickle


model = utils.__dict__['vgg16_bn']()
model.load_state_dict(torch.load('./cifar10_before.pth.tar'), strict=True)

print("model:\n", model.state_dict())

#print(model['features.0'])


"""
with open('mask.pickle', 'rb') as fr:
    mask = pickle.load(fr)

print(mask['features.0.weight'])
"""








