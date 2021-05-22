import os
import torch
import numpy as np
import math

def get_pattern(patterns, arr):               # input : (?, 1, 9) / output : (?, 10) 
    l = len(arr)

    for j in range(l):
        found_flag = 0
        for i in range(len(patterns)):
            if np.array_equal([patterns[i][0:9]], arr[j].tolist()):
                patterns[i][9] = patterns[i][9]+1
                found_flag = 1
                break;

        if(found_flag == 0):
            y = np.c_[arr[j], [1]]
            patterns.append(y.tolist()[0])
    return patterns    


def top_4(arr):                     # input : (d, ch, 1, 9) / output : (d*ch, 1, 9)
    print(arr.shape)
    arr = arr.reshape(-1,1,9)
    l = len(arr)

    for i in range(l):
        x = arr[i].copy()
        x.sort()
        arr[i]=np.where(arr[i]<x[0][5], 0, 1)

    return arr                     


def pattern_setter(model, num_sets=8):
    patterns = [[0,0,0,0,0,0,0,0,0,  0]]
    
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and name.split('.')[0] == "features" and len(param.shape) == 4:
            print(f'name:{name}')
            par=param.detach().numpy()
            patterns=get_pattern(patterns, top_4(par))
 
    patterns = np.array(patterns, dtype='int')
    patterns = patterns[patterns[:,9].argsort(kind='mergesort')]
    patterns = np.flipud(patterns)

    # print(patterns) 

    # print("top", str(num_sets), "patterns")
    pattern_set = patterns[:num_sets,:9]
    # print(pattern_set)
    
    return pattern_set

# new !!!!!!!!!  I wanna test it!
def top_4_pat(arr, pattern_set):    # input arr : (d, ch, 3, 3)   pattern_set : (6~8, 9) (9 is 3x3)
    cpy_arr = arr.copy().reshape(-1, 9)
    new_arr = np.zeros(cpy_arr.shape)
    pat_set = pattern_set.copy().reshape(-1, 9)

    for i in range(len(cpy_arr)):
        pat_arr = cpy_arr[i] * pat_set
        pat_arr = np.linalg.norm(pat_arr, axis=1)
        pat_idx = np.argmax(pat_arr)

        new_arr[i] = cpy_arr[i] * pat_set[pat_idx]

    new_arr = new_arr.reshape(arr.shape)
    return new_arr


def top_k_kernel(arr, perc):    # input (d, ch, 3, 3)
    k = math.ceil(arr.shape[0] * arr.shape[1] / perc)
    new_arr = arr.copy().reshape(-1, 9)    # (d*ch, 9)

    l2_arr = np.linalg.norm(new_arr, axis=1)
    threshold = l2_arr[np.argsort(-l2_arr)[k-1]]

    l2_arr = l2_arr >= threshold
    l2_arr = l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr
    l2_arr = np.transpose(np.array(l2_arr))
    
    new_arr = new_arr * l2_arr
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


##### for 'main_swp.py' #####
def top_4_pat_swp(arr, pattern_set):   # input arr : (d, ch, 3, 3)   pattern_set : (6~8, 9) (9 is 3x3)
    cpy_arr = arr.copy().reshape(len(arr), -1, 9)
    new_arr = np.zeros(cpy_arr.shape)
    pat_set = pattern_set.copy().reshape(-1, 9)
    pat_rst = np.zeros(len(pat_set))

    pat_arr = 0
    for i in range(len(cpy_arr)):
        for j in range(len(pat_set)):
            pat_arr = cpy_arr[i] * pat_set[j]
            pat_rst[j] = np.linalg.norm(pat_arr.reshape(-1))
        
        pat_idx = np.argmax(pat_rst)
        new_arr[i] = cpy_arr[i] * pat_set[pat_idx]

    new_arr = new_arr.reshape(arr.shape)
    return new_arr




""" my mistake1... should use tensor / torch calculation! (for speed)

def top_4_pat(arr, pattern_set):    # input arr : (d, ch, 3, 3)   pattern_set : (6~8, 9) (9 is 3x3)
    cpy_arr = arr.copy().reshape(-1, 1, 9)
    new_arr = np.zeros(cpy_arr.shape)

    for i in range(len(cpy_arr)):
        max = -1
        for j in range(len(pattern_set)):
            pat_arr = cpy_arr[i] * pattern_set[j]
            pat_l2 = np.linalg.norm(cpy_arr[i])
            
            if pat_l2 > max:
                max = pat_l2
                new_arr[i] = pat_arr
        
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


def top_k_kernel(arr, perc):    # input (d, ch, 3, 3)
    k = math.ceil(arr.shape[0] * arr.shape[1] / perc)
    new_arr = arr.copy().reshape(-1, 1, 9)    # (d*ch, 1, 9)
    l2_arr = np.zeros(len(new_arr))

    for i in range(len(new_arr)):
        l2_arr[i] = np.linalg.norm(new_arr[i]) 
        
    threshold = l2_arr[np.argsort(-l2_arr)[k-1]]    # top k-th l2-norm

    for i in range(len(new_arr)):
        new_arr[i] = new_arr[i] * (l2_arr[i] >= threshold)
    
    new_arr = new_arr.reshape(arr.shape)
    return new_arr
"""





