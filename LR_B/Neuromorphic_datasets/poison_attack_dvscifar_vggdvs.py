'''
val()函数测试模型精度时有问题，需要修改！！！！！ （已修改）
'''

import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

import attack
import data_loaders
from functions import *
from models import *
from spikingjelly.datasets.poisoned_dataset import create_backdoor_data_loader
from numpy import double
from utils_backdoor import train, val
from torch.cuda import amp

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers',default=2, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b',default=8, type=int,metavar='N',help='mini-batch size')
parser.add_argument('-b_test', default=8, type=int, help='batch size for test')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# file configuration
parser.add_argument('-data_dir', type=str, default='/home/jinlingxin/SNN/RGA/datasets/dvscifar/', help='root dir of DVS128 Gesture dataset')

# model configuration
parser.add_argument('-dataname', default='cifar10', type=str, help='dataset name', choices=['gesture', 'cifar10', 'mnist'])
parser.add_argument('-arch','--model',default='vggdvs',type=str,help='model')
parser.add_argument('-num_labels','--num_labels',default=10,type=int,help='number of labels(classes)')
parser.add_argument('-T','--time',default=10, type=int,metavar='N',help='snn simulation time, set 0 as ANN')
parser.add_argument('-tau','--tau',default=1., type=float,metavar='N',help='leaky constant')
parser.add_argument('-en', '--encode', default='constant', type=str, help='(constant/poisson)')
parser.add_argument('-dvs', '--dvs', default=True, type=bool, help='whether to use dvs data')
parser.add_argument('-init_s', '--init_s', default=64, type=int, help='init_s for dvs gesture(64), nmnist(34)')

# training configuration
parser.add_argument('--epochs',default=100,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.01,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-wd','--wd',default=5e-4, type=float,help='weight decay')
parser.add_argument('--optim', default='sgd',type=str,help='model optimizer')
parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')

# trigger & attack configuration
parser.add_argument('-pos', default='top-left', type=str, help='Position of the triggger')
parser.add_argument('-type', default=0, type=int, help='Type of the trigger, 0 static, 1 dynamic')
parser.add_argument('-trigger_label', default=1, type=int, help='The index of the trigger label')
parser.add_argument('-polarity', default=2, type=int, help='The polarity of the trigger', choices=[0, 1, 2])
parser.add_argument('-epsilon', default=0.2, type=double,help='Percentage of poisoned data')
parser.add_argument('-trigger_size', default=0.1, type=float, help='The size of the trigger as the percentage of the image size')

args = parser.parse_args()

def main():
    #>>>>>>>IMPORTANT<<<<<<<< Edit log_dir
    log_dir = '%sdvs-poisonbackdoor-checkpoints'% (args.dataname)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    poison_trainloader, clean_testloader, poison_testloader = create_backdoor_data_loader(args.dataname, args.trigger_label, args.epsilon, args.pos, 
                                                                                          args.type, args.trigger_size, args.b, args.b_test, args.time, 
                                                                                          device, args.data_dir, args.polarity)
    
    if 'cnndvs' in args.model.lower():
        model = CNNDVS(args.time, args.num_labels, args.tau, 2, init_s=args.init_s)
    elif 'vggdvs' in args.model.lower():
        model = VGGDVS_Backdoor(args.model.lower(), args.time, args.num_labels, args.tau)
    # elif 'vgg' in args.model.lower():
    #     model = VGG(args.model.lower(), args.time, args.num_labels, znorm, args.tau)
    # elif 'resnet17' in args.model.lower():
    #     model = ResNet17(args.time, args.tau, num_labels, znorm)
    # elif 'resnet19' in args.model.lower():
    #     model = ResNet19(args.time, args.tau, num_labels, znorm)
    else:
        raise AssertionError("model not supported")

    model.set_simulation_time(args.time)
    model.to(device)
    model.poisson = (args.encode.lower() == 'poisson')

    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0
    best_asr = 0

    # IMPORTANT<<<<<<<<<<<<< modifed
    identifier = args.dataname
    identifier += args.model
    identifier += '_T[%d]'%(args.time)
    identifier += '_tau[%.2f]'%(args.tau)
    identifier += '_lr[%.4f]'%(args.lr)
    identifier += '_ep[%.3f]'%(args.epsilon)
    identifier += '_size[%.2f]'%(args.trigger_size)
    identifier += '_poliaity[%d]'%(args.polarity)
    identifier += '_pos[%s]'%(args.pos)

    if args.encode == 'poisson':
        identifier += "_poisson"
    identifier += args.suffix

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    logger.info('before training!')
    test_acc_clean = val(model, clean_testloader, device, args.time)
    logger.info('Test clean acc={:.3f}\n'.format(test_acc_clean))
    test_asr_backdoor = val(model, poison_testloader, device, args.time)
    logger.info('Test poison asr={:.3f}\n'.format(test_asr_backdoor))
    logger.info('start training!')
    
    for epoch in range(args.epochs):# Train the model
        loss, acc = train(model, device, poison_trainloader, criterion, optimizer, args.time, True)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        scheduler.step()
        test_acc_clean = val(model, clean_testloader, device, args.time)
        logger.info('Test clean acc={:.3f}\n'.format(test_acc_clean))
        test_asr_backdoor = val(model, poison_testloader, device, args.time)
        logger.info('Test poison asr={:.3f}\n'.format(test_asr_backdoor))

        if best_acc < test_acc_clean:
            best_acc = test_acc_clean
            best_asr = test_asr_backdoor
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth'%(identifier)))

    logger.info('Best Test acc={:.3f}, best ASR = {:.3f}'.format(best_acc, best_asr))

if __name__ == "__main__":
    main()