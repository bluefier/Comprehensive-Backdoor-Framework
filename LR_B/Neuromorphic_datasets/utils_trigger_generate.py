import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

# from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
# from adversarialbox.train import adv_train, FGSM_train_rnd
# from adversarialbox.utils import to_var, pred_batch, test
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from time import time
from torch.utils.data.sampler import SubsetRandomSampler
# from adversarialbox.utils import to_var, pred_batch, test, \
#     attack_over_test_data
import random
from math import floor
import operator
import numpy as np
import copy
import matplotlib.pyplot as plt
import os

def get_pos(data, pos, trigger_size):
    
    print(data.shape)
    perturbed_data = copy.deepcopy(data)
    # Determining the location of trigger embedding
    frame = perturbed_data[0]
    width, height = frame.shape[2:]

    # Swap every samples to the target class
    # new_targets[perm] = trigger_label

    size_width = int(trigger_size * width)
    size_height = int(trigger_size * height)

    if pos == 'top-left':
        x_begin = 0
        x_end = size_width
        y_begin = 0
        y_end = size_height

    elif pos == 'top-right':
        x_begin = int(width - size_width)
        x_end = width
        y_begin = 0
        y_end = size_height

    elif pos == 'bottom-left':

        x_begin = 0
        x_end = size_width
        y_begin = int(height - size_height)
        y_end = height

    elif pos == 'bottom-right':
        x_begin = int(width - size_width)
        x_end = width
        y_begin = int(height - size_height)
        y_end = height

    elif pos == 'middle':
        x_begin = int((width - size_width) / 2)
        x_end = int((width + size_width) / 2)
        y_begin = int((height - size_height) / 2)
        y_end = int((height + size_height) / 2)

    elif pos == 'random':
        # Note that every sample gets the same (random) trigger position
        # We can easily implement random trigger position for each sample by using the following code
        ''' TODO:
            new_data[perm, :, np.random.randint(
            0, height, size=len(perm)), np.random.randint(0, width, size=(perm))] = value
        '''
        x_begin = np.random.randint(0, width)
        x_end = x_begin + size_width
        y_begin = np.random.randint(0, height)
        y_end = y_begin + size_height

    return x_begin, x_end, y_begin, y_end

# generating the trigger using fgsm method
class Trigger_Generation(object):

    def __init__(self, dataloader, criterion=None, gpu_id=0, epsilon=0.031, attack_method='pgd'):
        
        if criterion is not None:
            self.criterion =  nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()
            
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id #this is integer

        if attack_method is 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method is 'pgd':
            self.attack_method = self.pgd 
        
                                         
    def fgsm(self, model, data, target, tar, ep, x_begin, x_end, y_begin, y_end, data_min=0, data_max=1):
        
        model.eval()
        # perturbed_data = copy.deepcopy(data)
        perturbed_data = data.clone()
        
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        # print('output shape: ', output.shape)
        loss = self.criterion(output[:,tar], target[:,tar])
        # print(loss)

        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.mean().backward(retain_graph=True)
        
        # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False


        # Create the perturbed image by adjusting each pixel of the input image
        with torch.no_grad():
            # Create the perturbed image by adjusting each pixel of the input image
            # perturbed_data: [N, T, C, H, W]
            perturbed_data[:, :, 0:1, y_begin:y_end, x_begin:x_end] -= ep*sign_data_grad[:, :, 0:1, y_begin:y_end, x_begin:x_end]  ### 11X11 pixel would yield a TAP of 11.82 % 
            perturbed_data.clamp_(data_min, data_max) 
    
        # 返回触发器模式
        return perturbed_data
    