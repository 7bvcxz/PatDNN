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
import torchvision
from torchvision import datasets, transforms
import utils
from utils import *
from tqdm import tqdm


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
    best_prec = 0 
    for epoch in range(args.re_epoch):
        print('Re epoch: {}'.format(epoch+1))
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            #loss = regularized_nll_loss(args, model, F.log_softmax(output, dim=1), target)
            loss = regularized_nll_loss(args, model, output, target)
            loss.backward()
            optimizer.prune_step(mask)
        prec = test(args, model, device, test_loader)

        if prec > best_prec:
            best_prec = prec
            torch.save(model.state_dict(), os.path.join(args.save, 'cifar10_pruned.pth.tar'))


##### Settings #########################################################################
parser = argparse.ArgumentParser(description='Pytorch PatDNN training')
parser.add_argument('--model',      default='vgg16_bn',         help='select model')
parser.add_argument('--dir',        default='~/data',           help='data root')
parser.add_argument('--dataset',    default='cifar10',          help='select dataset')
parser.add_argument('--batchsize',  default=256, type=int,      help='set batch size')
parser.add_argument('--lr',         default=3e-4, type=float,   help='set learning rate')
parser.add_argument('--alpha',      default=5e-4, type=float,   help='set l2 regularization alpha')
parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='adam epsilon')
parser.add_argument('--rho',        default=1e-2, type=float,   help='set rho')
parser.add_argument('--connect_perc',  default=3.6, type=float, help='connectivity pruning ratio')
parser.add_argument('--epoch',      default=120, type=int,      help='set epochs')
parser.add_argument('--re_epoch',   default=20, type=int,       help='set retrain epochs')
parser.add_argument('--num_sets',   default='8', type=int,      help='# of pattern sets')
parser.add_argument('--exp',        default='test', type=str,   help='test or not')
parser.add_argument('--l2',         default=True, action='store_true',  help='apply l2 regularization')
parser.add_argument('--scratch',    default=False, action='store_true', help='start from pretrain/scratch')
parser.add_argument('--no-cuda',    default=False, action='store_true', help='disables CUDA training')
args = parser.parse_args()
print(args)
comment = "helloEveryone"

if args.exp == 'test':
    args.exp = f'{args.exp}-{time.strftime("%y%m%d-%H%M%S")}'
args.save = f'logs/{args.dataset}/{args.model}/{args.exp}_lr{str(args.lr)}_{comment}'
create_exp_dir(args.save)

args.workers = 16

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
########################################################################################


##### Load Pre-trained Model #####
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
history_score = np.zeros((args.epoch+1, 2))


# Optimizer
optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
# optimizer = optim.Adam(model.parameters(), args.lr)


print('\nTraining...') ##### ##### ##### ##### #####
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

    if prec > best_prec:
        best_prec = prec
        torch.save(model.state_dict(), os.path.join(args.save, 'cifar10_best.pth.tar'))



# Real Pruning ! ! !
print("\nApply Pruning with connectivity & pattern set...")
mask = apply_prune(args, model, device, pattern_set)
print_prune(model)

print("\ntesting...")
test(args, model, device, test_loader)

# Optimizer for Retrain
optimizer = PruneAdam(model.named_parameters(), lr=args.lr, eps=args.adam_epsilon)
# optimizer = optim.Adam(model.parameters(), args.lr)

print("\nfine-tuning...")
retrain(args, model, mask, device, train_loader, test_loader, optimizer)

prec = test(args, model, device, test_loader)
history_score[args.epoch][0] = 0
history_score[args.epoch][1] = prec
np.savetxt(os.path.join(args.save, 'train_record.txt'), history_score, fmt='%10.5f', delimiter=',')





