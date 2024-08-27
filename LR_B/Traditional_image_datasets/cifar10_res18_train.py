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
from spikingjelly.activation_based.model import spiking_resnet
from modify_dataset import build_poisoned_training_set, build_testset
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch

# 指定该程序在第二块GPU上运行
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    '''
    :return: None

    * :ref:`API in English <lif_fc_CIFAR10.main-en>`

    .. _lif_fc_CIFAR10.main-cn:

    使用全连接-LIF的网络结构，进行CIFAR10识别。\n
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。

    * :ref:`中文API <lif_fc_CIFAR10.main-cn>`

    .. _lif_fc_CIFAR10.main-en:

    The network with FC-LIF structure for classifying CIFAR10.\n
    This function initials the network, starts trainingand shows accuracy on test dataset.
    '''
    parser = argparse.ArgumentParser(description='LIF CIFAR10 Training')
    parser.add_argument('-T', default=8, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default= './data', type=str, help='root dir of CIFAR10 dataset')
    parser.add_argument('-out-dir', default='/home/jinlingxin/SNN/spikingjelly/spikingjelly/activation_based/examples/checkpoints', type=str, help='root dir for saving logs and checkpoint')
    # parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-tau', default=1.0, type=float, help='parameter tau of LIF neuron')
    parser.add_argument('-d', default='CIFAR10', help='use CIFAR10 dataset')
    parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
    parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')

    args = parser.parse_args()
    print(args)

    identifier = 'cifar10_res18_ori'
    net = spiking_resnet.spiking_resnet18(pretrained=False, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

    print(net)
    net.to(args.device)
    
   # 初始化数据加载器
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

    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     net.load_state_dict(checkpoint['net'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     max_test_acc = checkpoint['max_test_acc']
    
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')

    if args.amp:
        out_dir += '_amp'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    encoder = encoding.PoissonEncoder()

    for epoch in range(start_epoch, args.epochs):
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

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

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
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        print('*****test_acc*****', test_acc)
        print('*****max_test_acc*****', max_test_acc)

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
            print("****save path***:", os.path.join(out_dir, 'checkpoint_max_cifar10_res18.pth'))
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max_cifar10_res18.pth'))

        # torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest_lif_fc_CIFAR10.pth'))

        print(args)
        print(out_dir)
        print(f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    # 保存绘图用数据
    net.eval()
    # 注册钩子
    output_layer = net.layer[-1] # 输出层
    output_layer.v_seq = []
    output_layer.s_seq = []
    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)

    with torch.no_grad():
        img, label = test_dataset[0]
        img = img.to(args.device)
        out_fr = 0.
        for t in range(args.T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array_cifar10_res18_ori.npy",v_t_array)
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array_cifar10_res18_ori.npy",s_t_array)

    # 保存脉冲图
    torch.save(net.state_dict(), os.path.join(out_dir, '%s.pth'%(identifier)))
    

if __name__ == '__main__':
    train()
