'''
mode: STBP 
直接使用投毒攻击毒化snn模型，针对cifar10数据集和vgg1
'''

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
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch
from spikingjelly import visualizing
from matplotlib import pyplot as plt

# 指定该程序在第二块GPU上运行
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neuron_name = "IF"

def main():
    '''
    :return: None

    * :ref:`API in English <lif_fc_cifar10.main-en>`

    .. _lif_fc_cifar10.main-cn:

    使用全连接-LIF的网络结构，进行cifar10识别。\n
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。

    * :ref:`中文API <lif_fc_cifar10.main-cn>`

    .. _lif_fc_cifar10.main-en:

    The network with FC-LIF structure for classifying cifar10.\n
    This function initials the network, starts trainingand shows accuracy on test dataset.
    '''
    parser = argparse.ArgumentParser(description='LIF cifar10 Training')
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
    # parser.add_argument('-device', default='cuda:6', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
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
    # poison settings
    parser.add_argument('--poisoning_rate', type=float, default=0.2, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_10.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
    args = parser.parse_args()
    print(args)

    net = spiking_vgg.spiking_vgg11(pretrained=False, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

    # print(net)
    net.to(device)
    
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_test_clean, dataset_test_poisoned = build_testset(is_train=False, args=args)

    train_data_loader        = DataLoader(dataset_train, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)
    test_clean_data_loader   = DataLoader(dataset_test_clean, batch_size=args.b, shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)
    test_poisoned_data_loader = DataLoader(dataset_test_poisoned,  batch_size=args.b, shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True) # shuffle 随机化

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

    start_time = time.time()
    train_time = 0

    if args.load_local:
        print("## Load model from : %s" % args.out_dir)
        net.load_state_dict(torch.load(args.out_dir + 'basic_model_path'), strict=True)
        test_stats = evaluate_badnets(test_clean_data_loader, test_poisoned_data_loader, net, device)
        print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")
    else:
        print(f"Start training for {args.epochs} epochs")
        stats = []
        # test accuracy before training
        test_stats = evaluate_badnets(test_clean_data_loader, test_poisoned_data_loader, net, device, args.T, train_time)
        print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(train_data_loader, net, optimizer, args.loss, device, scaler, args.T, start_time)
            train_time = train_stats['train_time']
            test_stats = evaluate_badnets(test_clean_data_loader, test_poisoned_data_loader, net, device, args.T, train_time)
            print(f"# EPOCH {epoch}  Train loss: {train_stats['train_loss']:.4f} Train Acc: {train_stats['train_acc']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
            
            # save model 
            save_max = False
            if test_stats['clean_acc'] > max_test_acc:
                max_test_acc = test_stats['clean_acc']
                save_max = True
            
            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }

            if save_max:
                print("****save path***:", os.path.join(out_dir, 'checkpoint_max_cifar10_vgg11_%s.pth' % (neuron_name)))
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max_cifar10_vgg11_%s.pth' % (neuron_name)))

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
            }

            # save training stats
            stats.append(log_stats)
            df = pd.DataFrame(stats)
            df.to_csv("./logs/%s_trigger_vgg11_%d_%s_%.2f.csv" % (args.dataset, args.trigger_label, neuron_name, args.poisoning_rate), index=False, encoding='utf-8')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    # # 保存绘图用数据
    # net.eval()
    # # 注册钩子
    # # feature[1] (100, 64, 32, 32)
    # output_layer = net.features[1] 
    # output_layer.v_seq = []
    # output_layer.s_seq = []

    # def save_hook(m, x, y):
    #     m.v_seq.append(m.v.unsqueeze(0))
    #     m.s_seq.append(y.unsqueeze(0))

    # output_layer.register_forward_hook(save_hook)
    # loader_clean = DataLoader(dataset_test_clean, batch_size=1, shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)
    # loader_poison = DataLoader(dataset_test_poisoned, batch_size=1, shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)
    
    # # 神经元在干净测试集上的脉冲频率
    # with torch.no_grad():
    #     for t, (img, label) in enumerate(loader_clean):

    #         if t != 0:
    #             break
            
    #         img = img.to(device)
    #         out_fr = 0.
            
    #         for t in range(args.T):
    #             encoded_img = encoder(img)
    #             # print('draw shape: ', encoded_img.shape)
    #             out_fr += net(encoded_img)
    #         out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
    #         print(f'Firing rate-clean: {out_spikes_counter_frequency}')

    #         output_layer.v_seq = torch.cat(output_layer.v_seq)
    #         output_layer.s_seq = torch.cat(output_layer.s_seq)
    #         v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j]表示神经元i在j时刻的电压值
    #         v_t_array = v_t_array.reshape((100, 64*32*32))
    #         # np.save("v_t_array_clean_vgg11_cifar10_%s.npy" % (neuron_name), v_t_array)
    #         s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
    #         s_t_array = s_t_array.reshape((100, 64*32*32))
    #         # np.save("s_t_array_clean_vgg11_cifar10_%s.npy" % (neuron_name), s_t_array)
    #         print('1_v_shape: ', v_t_array.shape, '1_s_shape: ', s_t_array.shape)

    #         visualizing.plot_2d_heatmap(array=np.asarray(v_t_array), title='Membrane Potentials IF Neuron', xlabel='Simulating Step',
    #                             ylabel='Neuron Index', int_x_ticks=True, x_max=args.T, dpi=200)
    #         plt.show()

    #         visualizing.plot_1d_spikes(spikes=np.asarray(s_t_array), title='Membrane Potentials IF Neuron', xlabel='Simulating Step',
    #                         ylabel='Neuron Index', dpi=200)
    #         plt.show()


if __name__ == '__main__':
    main()
