from matplotlib.pyplot import sca
import torch.nn as nn
from torch import optim
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import seaborn as sns
import csv
from spikingjelly.clock_driven import functional
from torch.cuda import amp

def train(model, device, train_loader, criterion, optimizer, T, dvs):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        if dvs:
            images = images.transpose(0, 1)
        # print(images.type())
        # print(labels.shape)
        # print(labels)
        
        if T == 0:
            outputs = model(images)
        else:
            outputs = model(images).mean(0)
            # print('outputs shape: ', outputs.shape)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        # print(outputs.shape)
        # print(predicted.shape)
        # print('predicted shape: ', predicted.shape)
        # print('labels shape: ', labels.shape)
        correct += float(predicted.eq(torch.argmax(labels.cpu(), dim=1)).sum().item())

    return running_loss, 100 * correct / total

def val(model, test_loader, device, T):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        images = images.transpose(0, 1).float()
        # print(images.type())
        with torch.no_grad():
            if T > 0:
                outputs = model(images).mean(0)
            else:
                outputs = model(images)
        # print('output shape: ', outputs.shape)  
        _, predicted = outputs.cpu().max(1)
        # print('predicted shape: ', predicted.shape)
        # print('labels shape: ', labels.shape)
        total += float(labels.size(0))
        correct += float(predicted.eq(torch.argmax(labels.cpu(), dim=1)).sum().item())
    final_acc = 100 * correct / total

    return final_acc

def evaluate(model, test_loader, criterion, device):
    '''
    Evaluate the model on the test set
    Parameters:
        model (torch.nn.Module): model
        test_loader (torch.utils.data.DataLoader): test loader
        criterion (torch.nn.modules.loss._Loss): loss function
        device (torch.device): device
    Returns:
        test_loss (float): test loss
        test_acc (float): test accuracy
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device), target.to(device)
            output = model(data).mean(0)
            # print(output.shape)
            # print(target.shape)
            test_loss += criterion(output, target).item()
            # pred = output.argmax(dim=1)
            # correct += pred.eq(torch.argmax(target, dim=1)).sum().item()
            _, predicted = output.cpu().max(1)
            # print(outputs.shape)
            # print(predicted.shape)
            correct += float(predicted.eq(torch.argmax(target.cpu(), dim=1)).sum().item())
            functional.reset_net(model)

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    return test_loss, test_acc
