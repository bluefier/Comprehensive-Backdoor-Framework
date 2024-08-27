'''
mode: ANN2SNN
step 1：训练trojaned ANN模型
'''

import argparse
import os
import pathlib
import re
import time
import datetime
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
import cifar10_vgg
from modify_dataset import build_poisoned_training_set_time, build_testset_time
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch_ann, evaluate_badnets_ann
import spikingjelly.activation_based.ann2snn as ann2snn

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=4, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_dir', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
parser.add_argument('--log', action='store_true', help='to print the output on terminal or to log file')
# poison settings
# parser.add_argument('--poisoning_rate', type=float, default=0.01, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="/home/jinlingxin/SNN/spikingjelly/spikingjelly/activation_based/examples/triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

args = parser.parse_args()

log_file = '/home/jinlingxin/SNN/spikingjelly/pretraind_model/trojaned_vgg11/'
try:
    os.mkdir(log_file)
except OSError:
    pass 

identifier = args.dataset.lower()
log_file += identifier+'_whole_1.log'

if args.log:
        f = open(log_file, 'w', buffering=1)
        print('=================================open file========================================')
else:
        f = sys.stdout

pr_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

def main():
    print("{}".format(args).replace(', ', ',\n'))

    # # create related path
    # pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    # pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)
    for pr in pr_list:
        f.write("\n# Poisoning Rate: %f " % pr)
        f.write("\n# load dataset: %s " % args.dataset)
        dataset_train, args.nb_classes = build_poisoned_training_set_time(pr, is_train=True, args=args)
        dataset_val_clean, dataset_val_poisoned = build_testset_time(pr, is_train=False, args=args)
        
        data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # shuffle 随机化

        model = cifar10_vgg.VGG("VGG11").to(device)
        model_ann2snn = cifar10_vgg.VGG("VGG11").to(device)
        # model = VGG('VGG11').to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)
        model_structure = 'VGG11'
        basic_model_path = "/home/jinlingxin/SNN/spikingjelly/pretraind_model/pa_%s_%s_trojaned_ann_%.2f.pth" % (args.dataset, model_structure, pr)
        
        max_acc = 0
        # max_asr = 0
        T = 400

        start_time = datetime.datetime.now()
        for epoch in range(args.epochs):
            train_stats = train_one_epoch_ann(data_loader_train, model, criterion, optimizer, args.loss, device)
            test_stats = evaluate_badnets_ann(data_loader_val_clean, data_loader_val_poisoned, model, device)
            print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
            
            if test_stats['clean_acc'] > max_acc:
                max_acc = test_stats['clean_acc']
                # save model 
                print('---save model---')
                torch.save(model.state_dict(), basic_model_path)
                f.write(f"\n# EPOCH {epoch} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}")
                f.write('\n cost time: {}'.format(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))  

        model_ann2snn.load_state_dict(torch.load('/home/jinlingxin/SNN/spikingjelly/pretraind_model/pa_%s_%s_trojaned_ann_%.2f.pth' % (args.dataset, model_structure, pr)))
        f.write('\n---------------------------------------------')
        f.write('\nConverting using MaxNorm')
        start_time = datetime.datetime.now()
        model_converter = ann2snn.Converter(mode='Max', dataloader=data_loader_train)
        snn_model = model_converter(model_ann2snn)
        f.write('\n cost time: {}'.format(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))  
        f.write('\n SNN accuracy:')
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, snn_model, device, T, start_time)
        f.write('\nACC: {}'.format(test_stats['clean_acc']))
        f.write('\nASR: {}'.format(test_stats['asr']))
        # snn_acc_max.append(test_stats['clean_acc'])
        # snn_asr_max.append(test_stats['asr'])
        f.write('\n---------------------------------------------')
        f.write('\nConverting using RobustNorm')
        start_time = datetime.datetime.now()
        model_converter = ann2snn.Converter(mode='99.9%', dataloader=data_loader_train)
        snn_model = model_converter(model_ann2snn)
        f.write('\n cost time: {}'.format(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))
        f.write('\nSNN accuracy:')
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, snn_model, device, T, start_time)
        f.write('\nACC: {}'.format(test_stats['clean_acc']))
        f.write('\nASR: {}'.format(test_stats['asr']))
        # snn_acc_99.append(test_stats['clean_acc'])
        # snn_asr_99.append(test_stats['asr'])
        f.write('\n---------------------------------------------')
        f.write('\nConverting using 1/2 max(activation) as scales...')
        start_time = datetime.datetime.now()
        model_converter = ann2snn.Converter(mode=1.0 / 2, dataloader=data_loader_train)
        snn_model = model_converter(model_ann2snn)
        f.write('\n cost time: {}'.format(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))
        f.write('\nSNN accuracy:')
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, snn_model, device, T, start_time)
        # snn_acc_1_2.append(test_stats['clean_acc'])
        # snn_asr_1_2.append(test_stats['asr'])
        f.write('\nACC: {}'.format(test_stats['clean_acc']))
        f.write('\nASR: {}'.format(test_stats['asr']))
        f.write('\n---------------------------------------------')
        f.write('\nConverting using 1/3 max(activation) as scales...')
        start_time = datetime.datetime.now()
        model_converter = ann2snn.Converter(mode=1.0 / 3, dataloader=data_loader_train)
        snn_model = model_converter(model_ann2snn)
        f.write('\n cost time: {}'.format(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))
        f.write('\nSNN accuracy:')
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, snn_model, device, T, start_time)
        f.write('\nACC: {}'.format(test_stats['clean_acc']))
        f.write('\nASR: {}'.format(test_stats['asr']))
        # snn_acc_1_3.append(test_stats['clean_acc'])
        # snn_asr_1_3.append(test_stats['asr'])
        f.write('\n---------------------------------------------')
        f.write('\nConverting using 1/4 max(activation) as scales...')
        start_time = datetime.datetime.now()
        model_converter = ann2snn.Converter(mode=1.0 / 4, dataloader=data_loader_train)
        snn_model = model_converter(model_ann2snn)
        f.write('\n cost time: {}'.format(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))
        f.write('\nSNN accuracy:')
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, snn_model, device, T, start_time)
        f.write('\nACC: {}'.format(test_stats['clean_acc']))
        f.write('\nASR: {}'.format(test_stats['asr']))
        # snn_acc_1_4.append(test_stats['clean_acc'])
        # snn_asr_1_4.append(test_stats['asr'])
        f.write('\n--------------------------------------------')
        f.write('\nConverting using 1/5 max(activation) as scales...')
        start_time = datetime.datetime.now()
        model_converter = ann2snn.Converter(mode=1.0 / 5, dataloader=data_loader_train)
        snn_model = model_converter(model_ann2snn)
        f.write('\n cost time: {}'.format(datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)))
        f.write('\nSNN accuracy:')
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, snn_model, device, T, start_time)
        f.write('\nACC: {}'.format(test_stats['clean_acc']))
        f.write('\nASR: {}'.format(test_stats['asr']))   

if __name__ == "__main__":
    main()