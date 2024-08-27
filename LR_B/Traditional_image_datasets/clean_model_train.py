'''
train clean model for different settings
'''

import argparse
import os
import pathlib
import re
import time
import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from models import *
from modify_dataset import build_poisoned_training_set, build_testset
from deeplearning_ann import evaluate_badnets, optimizer_picker, train_one_epoch

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=200, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=4, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_dir', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

args = parser.parse_args()

def main():
    print("{}".format(args).replace(', ', ',\n'))

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("\n# load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)
    
    data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # shuffle 随机化

    model = ResNet18().to(device)
    # model = VGG('VGG11').to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)
    model_structure = 'res18'
    basic_model_path = "/home/jinlingxin/SNN/spikingjelly/pretraind_model/clean_%s_%s.pth" % (args.dataset, model_structure)
    start_time = time.time()
    max_acc = 0

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, args.loss, device)
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
        print(f"# EPOCH {epoch}  loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}\n")
        
        # save model 
        if test_stats['clean_acc'] > max_acc:
            print('saving.....')
            torch.save(model.state_dict(), basic_model_path)
            max_acc = test_stats['clean_acc']

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()