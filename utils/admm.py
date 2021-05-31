import torch
import torch.nn.functional as F
import numpy as np
from .pattern_setter import *

def regularized_nll_loss(args, model, output, target):
    loss = F.cross_entropy(output, target)
    return loss

def admm_loss(args, device, model, Z, Y, U, V, output, target):
    idx = 0
    loss = F.cross_entropy(output, target)
    
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name:
            z = Z[idx].to(device)
            y = Y[idx].to(device)
            u = U[idx].to(device)
            v = V[idx].to(device)

            loss += args.rho * 0.5 * (param - z + u).norm() + args.rho * 0.5 * (param - y + v).norm()
            idx += 1
    return loss


def initialize_Z_Y_U_V(model):
    Z = ()
    Y = ()
    U = ()
    V = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name:
            Z += (param.detach().cpu().clone(),)
            Y += (param.detach().cpu().clone(),)
            U += (torch.zeros_like(param).cpu(),)
            V += (torch.zeros_like(param).cpu(),)
    return Z, Y, U, V


def update_X(model):
    X = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name:
            X += (param.detach().cpu().clone(),)
    return X


def update_Z(X, U, pattern_set, args):
    new_Z = ()
    for x, u in zip(X, U):
        z = x + u
        
        # Select each kernel and prune -> z = torch.tensor (a, b, 3, 3) or (a, b, 1, 1)
        z = torch.from_numpy(top_4_pat(z.numpy(), pattern_set))

        new_Z += (z,)
    return new_Z


def update_Y(X, V, args):
    new_Y = ()
    for x, v in zip(X, V):
        y = x + v

        # Prune kernel by l2 -> y = torch.tensor (a, b, 3, 3) or (a, b, 1, 1) 
        y = torch.from_numpy(top_k_kernel(y.numpy(), args.connect_perc)) 

        new_Y += (y,)
    return new_Y


def update_U(U, X, Z):
    new_U = ()
    for u, x, z in zip(U, X, Z):
        new_u = u + x - z
        new_U += (new_u,)
    return new_U


def update_V(V, X, Y):
    new_V = ()
    for v, x, y in zip(V, X, Y):
        new_v = v + x - y
        new_V += (new_v,)
    return new_V


def prune_weight(weight, device, percent, pattern_set):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()

    weight_numpy = top_k_kernel(weight_numpy, percent)
    weight_numpy = top_4_pat(weight_numpy, pattern_set)
    
    mask = torch.Tensor(weight_numpy != 0).to(device)
    return mask

def apply_prune(args, model, device, pattern_set):
    # returns dictionary of non_zero_values' indices
    dict_mask = {}
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name:
            mask = prune_weight(param, device, args.connect_perc, pattern_set)
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
    return dict_mask

    
def print_convergence(model, X, Z):
    idx = 0
    print("\nnormalized norm of (weight - projection)")
    for name, _ in model.named_parameters():
        if name.split('.')[-1] == "weight":
            x, z = X[idx], Z[idx]
            print("({}): {:.4f}".format(name, (x-z).norm().item() / x.norm().item()))
            idx += 1


def print_prune(model):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))


##### for 'main_swp.py' #####
def update_Z_swp(X, U, pattern_set, args):
    new_Z = ()
    for x, u in zip(X, U):
        z = x + u
        
        # Select each kernel and prune -> z = torch.tensor (a, b, 3, 3)
        z = torch.from_numpy(top_4_pat_swp(z.numpy(), pattern_set))

        new_Z += (z,)
    return new_Z

def apply_prune_swp(args, model, device, pattern_set):
    # returns dictionary of non_zero_values' indices
    dict_mask = {}
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name:
            mask = prune_weight_swp(param, device, args.connect_perc, pattern_set)
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
    return dict_mask

def prune_weight_swp(weight, device, percent, pattern_set):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()

    weight_numpy = top_k_kernel(weight_numpy, percent)
    weight_numpy = top_4_pat_swp(weight_numpy, pattern_set)
    
    mask = torch.Tensor(weight_numpy != 0).to(device)
    return mask


##### for 'main_loss3.py test' #####
def admm_lossc(args, device, model, Z, Y, U, V, output, target):
    loss = F.cross_entropy(output, target)
    return loss

def admm_lossz(args, device, model, Z, Y, U, V, output, target):
    idx = 0
    loss = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name:
            z = Z[idx].to(device)
            u = U[idx].to(device)

            loss += args.rho * 0.5 * (param - z + u).norm()
            idx += 1
    return loss

def admm_lossy(args, device, model, Z, Y, U, V, output, target):
    idx = 0
    loss = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and len(param.shape) == 4 and 'downsample' not in name:
            y = Y[idx].to(device)
            v = V[idx].to(device)

            loss += args.rho * 0.5 * (param - y + v).norm()
            idx += 1
    return loss


