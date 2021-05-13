from __future__ import print_function
import os
import argparse
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
from torchvision import datasets, transforms
import utils
from utils import *
from tqdm import tqdm
import pickle

##### train, test, retrain function #######################################################
def train(args, model, device, pattern_set, train_loader, test_loader, optimizer, Z, Y, U, V):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = admm_loss(args, device, model, Z, Y, U, V, F.log_softmax(output, dim=1), target)
        loss = admm_loss(args, device, model, Z, Y, U, V, output, target)
        loss.backward()
        optimizer.step()
       
    X = update_X(model)

    print("update Z, Y, U, V...")
    Z = update_Z(X, U, pattern_set, args)
    print("updated Z")
    Y = update_Y(X, V, args)
    print("updated Y")

    U = update_U(U, X, Z)
    V = update_V(V, X, Y)
    print("updated U V")


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    prec = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), prec))

    return prec


def retrain(args, model, mask, device, train_loader, test_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = regularized_nll_loss(args, model, F.log_softmax(output, dim=1), target)
        loss = regularized_nll_loss(args, model, output, target)
        loss.backward()
        optimizer.prune_step(mask)


##### Settings #########################################################################
parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--model',      default='vgg16_bn',         help='select model')
parser.add_argument('--dir',        default='~/data',           help='dataset root')
parser.add_argument('--dataset',    default='cifar10',          help='select dataset')
parser.add_argument('--batchsize',  default=256, type=int,      help='set batch size')
parser.add_argument('--lr',         default=3e-5, type=float,   help='set learning rate')
parser.add_argument('--re_lr',      default=1e-4, type=float,   help='set fine learning rate')
parser.add_argument('--alpha',      default=5e-4, type=float,   help='set l2 regularization alpha')
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='adam epsilon')
parser.add_argument('--rho',        default=1e-2, type=float,   help='set rho')
parser.add_argument('--connect_perc',  default=3.6, type=float, help='connectivity pruning ratio')
parser.add_argument('--epoch',      default=120, type=int,      help='set epochs')
parser.add_argument('--re_epoch',   default=60, type=int,       help='set retrain epochs')
parser.add_argument('--num_sets',   default='8', type=int,      help='# of pattern sets')
parser.add_argument('--exp',        default='test', type=str,   help='test or not')
parser.add_argument('--l2',         default=False, action='store_true', help='apply l3 regularization')
parser.add_argument('--scratch',    default=False, action='store_true', help='start from pretrain/scratch')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
args = parser.parse_args()
print(args)
comment = "check6"

if args.exp == 'test':
    args.exp = f'{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
args.save = f'logs/{args.dataset}/{args.model}/{args.exp}_lr{str(args.lr)}_rls{str(args.re_lr)}_{comment}'

args.workers = 16

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
##########################################################################################################

print('Preparing pre-trained model...')
if args.dataset == 'imagenet':
    pre_model = torchvision.models.vgg16(pretrained=True)
elif args.dataset == 'cifar10':
    pre_model = utils.__dict__[args.model]()
    pre_model.load_state_dict(torch.load('./cifar10_pretrain/vgg16_bn.pt'), strict=True)
print("pre-trained model:\n", pre_model)


##### Find Pattern Set #####
print('\nFinding Pattern Set...')
if os.path.isfile('pattern_set_'+ args.dataset + '.npy') is False:
    pattern_set = pattern_setter(pre_model)
    np.save('pattern_set_'+ args.dataset + '.npy', pattern_set)
else:
    pattern_set = np.load('pattern_set_'+ args.dataset + '.npy')

pattern_set = pattern_set[:args.num_sets, :]
print(pattern_set)
print('pattern_set loaded')


##### Load Dataset ####
print('\nPreparing Dataset...')
train_loader, test_loader = data_loader(args.dir, args.dataset, args.batchsize, args.workers)
print('Dataset Loaded')


##### Load Model #####
model = utils.__dict__[args.model]()

# if pre-trained... load pre-trained weight
if not args.scratch:
    state_dict = pre_model.state_dict()
    torch.save(state_dict, 'tmp_pretrained.pt')

    model.load_state_dict(torch.load('tmp_pretrained.pt'), strict=True)
model.cuda()
pre_model.cuda()


# History collector
history_score = np.zeros((args.epoch+args.re_epoch+1, 2))


# Optimizer
optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)


print('\nTraining...') ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
best_prec = 0
Z, Y, U, V = initialize_Z_Y_U_V(model)

print('Pre_model:')
test(args, pre_model, device, test_loader)
print('Our_model:')
test(args, model, device, test_loader)

for epoch in range(args.epoch):
    print("Epoch: {} with lr: {}".format(epoch+1, args.lr))
    if epoch in [args.epoch//4, args.epoch//2, args.epoch//4*3]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    
    train(args, model, device, pattern_set, train_loader, test_loader, optimizer, Z, Y, U, V)

    print("\ntesting...")
    prec = test(args, model, device, test_loader)
    history_score[epoch][0] = epoch
    history_score[epoch][1] = prec
    
create_exp_dir(args.save)
torch.save(model.state_dict(), os.path.join(args.save, 'cifar10_before.pth.tar'))


# Real Pruning ! ! !
print("\nApply Pruning with connectivity & pattern set...")
mask = apply_prune(args, model, device, pattern_set)
print_prune(model)

for name, param in model.named_parameters():
    if name.split('.')[-1] == "weight" and len(param.shape)==4:
        param.data.mul_(mask[name])

torch.save(model.state_dict(), os.path.join(args.save, 'cifar10_after.pth.tar'))

print("\ntesting...")
test(args, model, device, test_loader)


# Optimizer for Retrain
optimizer = PruneAdam(model.named_parameters(), lr=args.re_lr, eps=args.adam_epsilon)


# Fine-tuning...
print("\nfine-tuning...")
best_prec = 0
for epoch in range(args.re_epoch):
    print("Epoch: {} with re_lr: {}".format(epoch+1, args.re_lr))
    if epoch in [args.re_epoch//4, args.re_epoch//2, args.re_epoch//4*3]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    retrain(args, model, mask, device, train_loader, test_loader, optimizer)

    print("\ntesting...")
    prec = test(args, model, device, test_loader)
    history_score[args.epoch + epoch][0] = epoch
    history_score[args.epoch + epoch][1] = prec
    
    if prec > best_prec:
        best_prec = prec
        torch.save(model.state_dict(), os.path.join(args.save, 'cifar10_pruned.pth.tar'))

np.savetxt(os.path.join(args.save, 'train_record.txt'), history_score, fmt='%10.5f', delimiter=',')


############################################

# my mistake 1 - making mask.pickle
"""
with open('mask.pickle', 'wb') as fw:
    pickle.dump(mask, fw)

with open('mask.pickle', 'rb') as fr:
    mask = pickle.load(fr)
    print("mask loaded")
"""

# my mistake 2
"""
for module in model.named_modules():
    if isinstance(module[1], nn.Conv2d):
        print("module:", module[0])
        prune.custom_from_mask(module, 'weight', mask=mask[module[0] +'.weight'])
"""






