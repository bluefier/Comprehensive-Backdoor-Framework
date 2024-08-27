#############################
#   @author: Nitin Rathi    #
#############################
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import sys
import datetime
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchviz import make_dot
from modify_dataset import build_poisoned_training_set, build_testset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from self_models import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def train(epoch, loader):

    global learning_rate
    
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']
    
    #total_correct   = 0
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        
        start_time = datetime.datetime.now()

        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
                
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        #make_dot(loss).view()
        #exit(0)
        loss.backward()
        optimizer.step()
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        #total_correct += correct.item()

        losses.update(loss.item(), data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))
        
    f.write('\n Epoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
            epoch,
            learning_rate,
            losses.avg,
            top1.avg
            )
        )

def test(data_loader_val_clean, data_loader_val_poisoned, model, device):
    global max_accuracy, max_asr, start_time

    acc = eval(data_loader_val_clean, model, device, print_perform=True)
    asr = eval(data_loader_val_poisoned, model, device, print_perform=False)
    # if epoch > 30 and top1.avg<0.15:
    #     f.write('\n Quitting as the training is not progressing')
    #     exit(0)

    if acc['acc'] > max_accuracy:
        max_accuracy = acc['acc']
        max_asr = asr['acc']
        state = {
                'accuracy'      : max_accuracy,
                'epoch'         : epoch,
                'state_dict'    : model.state_dict(),
                'optimizer'     : optimizer.state_dict()
        }
        try:
            os.mkdir('./trained_models/ann_backdoor/')
        except OSError:
            pass
        
        filename = './trained_models/ann_backdoor/'+identifier+'.pth'
        torch.save(state,filename)
        
    f.write(' Test_acc: {:.4f}, best_acc: {:.4f}, Test_asr: {:.4f}, best_asr: {:.4f}, time: {}'.  format(
        acc['acc'], 
        max_accuracy,
        asr['acc'],
        max_asr,
        datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
        )
    )
    return {
            'clean_acc': acc['acc'], 'clean_loss': acc['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            }

def eval(data_loader, model, device, batch_size=64, print_perform=False):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')  

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = F.cross_entropy(output,target)
            total_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            losses.update(loss.item(), data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))
    
    return {
            "acc": top1.avg,
            "loss": losses.avg,
            }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ANN to be later converted to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--devices',                default='5',                type=str,       help='list of gpu device(s)')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='RESNET20',            type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    parser.add_argument('-lr','--learning_rate',    default=1e-2,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained model to initialize ANN')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--epochs',                 default=100,                type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--optimizer',              default='SGD',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')
    parser.add_argument('--momentum',               default=0.9,                type=float,     help='momentum parameter for the SGD optimizer')
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.2,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--data_dir',               default='./Datasets/cifar_data',            help='Place to load dataset (default: ./dataset/)')
    parser.add_argument('--nb_classes',             default=10,                 type=int,       help='number of the classification types')
    
    # poison settings
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

    args=parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
    device          = args.devices
    dataset         = args.dataset
    batch_size      = args.batch_size
    architecture    = args.architecture
    learning_rate   = args.learning_rate
    pretrained_ann  = args.pretrained_ann
    epochs          = args.epochs
    lr_reduce       = args.lr_reduce
    optimizer       = args.optimizer
    weight_decay    = args.weight_decay
    momentum        = args.momentum
    amsgrad         = args.amsgrad
    dropout         = args.dropout
    kernel_size     = args.kernel_size
    poison_rate     = args.poisoning_rate

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))
    
    
    log_file = './logs/ann_backdoor/'
    try:
        os.mkdir(log_file)
        print('create successfully!')
    except OSError:
        pass 
    
    #identifier = 'ann_'+architecture.lower()+'_'+dataset.lower()+'_'+str(datetime.datetime.now())
    identifier = 'ann_backdoor'+architecture.lower()+'_'+dataset.lower()+'_'+str(dropout)+'_'+str(poison_rate)+'_bottom-left'
    log_file+=identifier+'.log'
    
    if args.log:
        f= open(log_file, 'w', buffering=1)
    else:
        f=sys.stdout
    
    f.write('\n Run on time: {}'.format(datetime.datetime.now()))
            
    f.write('\n\n Arguments:')
    for arg in vars(args):
        if arg == 'lr_interval':
            f.write('\n\t {:20} : {}'.format(arg, lr_interval))
        else:
            f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
        
    # Training settings
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Loading Dataset
    if dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        labels      = 100 
    elif dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        labels      = 10
    elif dataset == 'MNIST':
        labels = 10
    
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_test_clean, dataset_test_poisoned = build_testset(is_train=False, args=args)

    train_data_loader        = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=torch.Generator(device = 'cuda'))
    test_clean_data_loader   = DataLoader(dataset_test_clean, batch_size=args.batch_size, shuffle=False, drop_last=False, generator=torch.Generator(device = 'cuda'))
    test_poisoned_data_loader = DataLoader(dataset_test_poisoned,  batch_size=args.batch_size, shuffle=False, drop_last=False, generator=torch.Generator(device = 'cuda')) # shuffle 随机化
    
    if architecture[0:3].lower() == 'vgg':
        model = VGG(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout)
    elif architecture[0:3].lower() == 'res':
        if architecture.lower() == 'resnet12':
            model = ResNet12(labels=labels, dropout=dropout)
        elif architecture.lower() == 'resnet20':
            model = ResNet20(labels=labels, dropout=dropout)
        elif architecture.lower() == 'resnet34':
            model = ResNet34(labels=labels, dropout=dropout) 
    #f.write('\n{}'.format(model))
    
    #CIFAR100 sometimes has problem to start training
    #One solution is to train for CIFAR10 with same architecture
    #Load the CIFAR10 trained model except the final layer weights
    model = nn.DataParallel(model)
    if args.pretrained_ann:
        state=torch.load(args.pretrained_ann, map_location='cpu')
        cur_dict = model.state_dict()
        for key in state['state_dict'].keys():
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                    f.write('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
                else:
                    f.write('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
            else:
                f.write('\n Error: Loaded weight {} not present in current model'.format(key))
        
        model.load_state_dict(cur_dict)
    
    f.write('\n {}'.format(model)) 
    
    # if torch.cuda.is_available() and args.gpu:
    #     model.cuda()
    
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay)
    
    f.write('\n {}'.format(optimizer))
    max_accuracy = 0
    max_asr = 0
    
    model = model.cuda()
    for epoch in range(1, epochs):
        
        start_time = datetime.datetime.now()
        train(epoch, train_data_loader)
        test_stats = test(test_clean_data_loader, test_poisoned_data_loader, model, device)
           
    f.write('\n Highest accuracy: {:.4f} highest ASR: {:.4f}'.format(max_accuracy, max_asr))
