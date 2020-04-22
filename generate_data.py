#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt 
from dataLoader import load_dataset, load_dataloader
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import os

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

def create_sum(dataset, l): 
    add = 0
    c = dataset.classes[l]
    digits = [int(d) for d in str(c)]
    for x in digits: 
        add += x
    return add



def gather_examples(loader, ds, filename):
    get_sums = list() 
    get_idx = list() 
    get_labels = list() 
    c = 0 
    for idx, (images, labels) in enumerate(loader):
        l = int(labels)
#        if c < 10: print("c: ", c)
        if c == 0: 
            print("label: ",labels, "idx: ", idx)
            plt.imshow(images[0,0,:,:])
        get_labels.append(l)
        sums = create_sum(ds,l)
        get_sums.append(sums)
        get_idx.append(idx)
        c += 1
    i = list(range(c))
    random.shuffle(i)
    with open(filename, 'w') as f: 
        for x in range(c):
            nr = i[x]
            if get_idx[nr] == 5440:
                plt.imshow(images[0,0,:,:])
                print("get index: {}, get_sums: {}, get labels: {}".format(get_idx[nr],get_sums[nr],get_labels[nr]))
            f.write("addition({},{}).\n".format(get_idx[nr],int(get_sums[nr])))

#def gather_examples(loader, filename):
#    get_idx = list() 
#    get_labels = list() 
#    c = 0 
#    for images, l in loader:
##        l = int(labels)
##        sums = create_sum(l)
#        get_idx.append(c)
#        get_labels.append(l)
#        c += 1
#    i = list(range(c))
##    print("len i : ", len(i))
#    random.shuffle(i)
##    print("c: ", c)
#    with open(filename, 'w') as f: 
#        for x in range(c):
#            nr = i[x]
##            print("nr: ",nr)
##            print("get_idx: ", len(get_idx))
#            f.write("addition({},{}).\n".format(get_idx[nr],int(get_labels[nr])))
            
            
train_path = "/root/Desktop/Original/deepproblog/data/train"
test_path = "/root/Desktop/Original/deepproblog/data/test"

train_set, train_loader = load_dataloader(train_path)
test_set, test_loader = load_dataloader(test_path)

#PATH ="/root/Desktop/Original/deepproblog/data/dataset/" #normal size 
#train_set, test_set = load_dataset(PATH)
#dataset, train_loader, test_loader = load_dataloader(PATH)
#
#gather_examples(train_loader,train_set,"train_dataN.txt")
#gather_examples(test_loader,test_set,"test_dataN.txt")


x = 4474
image, labels = train_set[x]
print(image.shape)
print(labels)

npimages = image.numpy()
npimages = np.transpose(image, (1,2,0))

plot = plt.imshow(npimages[:,:,0])
plt.show() 
