'''
mode: ANN2SNN-I
step 1: 将clean ANN模型(达到基础精度要求)转换为clean SNN模型
'''

# 加载python库，包含数据处理与pytorch自己的包
import os
import time
import argparse
import sys
import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from spikingjelly.activation_based.model import spiking_vgg
from modify_dataset import build_poisoned_training_set, build_testset
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

def main():
    # 选定运行设备（CPU或者GPU）
    # 如果GPU可用，那么指定第0,1,2...个GPU作为运行设备
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description='LIF cifar10 Training')
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default= './data', type=str, help='root dir of cifar10 dataset')
    parser.add_argument('-out-dir', default='/home/jinlingxin/SNN/spikingjelly/spikingjelly/activation_based/examples/checkpoints', type=str, help='root dir for saving logs and checkpoint')
    # parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    parser.add_argument('-d', default='CIFAR10', help='use cifar10 dataset')
    parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
    parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
    parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (cifar10 or CIFAR10, default: cifar10)')
    parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
    args = parser.parse_args()

    '''
    数据预处理部分
    '''
    print('==> Preparing data..')
    train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    print('==> Building model..')
    neuron_name = 'IFNode'
    model_name = 'spiking_vgg11'
    net = spiking_vgg.spiking_vgg11(pretrained=False, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    net.to(args.device)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')

    encoder = encoding.PoissonEncoder()

    for epoch in range(start_epoch, args.epochs):
        # print(f'====== epoch {epoch} begin ======')
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = 0.
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr / args.T
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        # print("*train_loss*:  ", train_loss,' *train_acc*: ', train_acc, ' *train_speed*: ', train_speed)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        # begin test
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples

        save_max = False
        # print('*test_acc*: ', test_acc)
        # print('*max_test_acc*: ', max_test_acc)

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            print("****save path***:", os.path.join(out_dir, 'checkpoint_%s_%s_%s.pth' % (args.dataset, model_name, neuron_name)))
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_%s_%s_%s.pth' % (args.dataset, model_name, neuron_name)))
        # torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest_lif_fc_mnist.pth'))

        # print(args)
        # print(out_dir)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


# python 程序主入口
if __name__ == "__main__":
    # 训练100个epoch（轮次）
    main()
