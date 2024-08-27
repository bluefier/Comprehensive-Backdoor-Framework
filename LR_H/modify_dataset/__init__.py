from .poisoned_dataset import CIFAR10Poison, MNISTPoison
from torchvision import datasets, transforms
import torch 
import os 

def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data

def build_poisoned_training_set(is_train, args):
    # transform, detransform = build_transform(args.dataset)
    # transform = build_transform(args.dataset)
    # print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        trans_train, trans_test = build_trans_CIFAR(args.dataset)
        print("Transform = ", trans_train)
        trainset = CIFAR10Poison(args, args.data_dir, train=is_train, download=True, transform=trans_train)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        transform = build_trans_MNIST(args.dataset)
        print("Transform = ", transform)
        trainset = MNISTPoison(args, args.data_dir, train=is_train, download=True, transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes

def build_testset(is_train, args):
    # transform, detransform = build_transform(args.dataset)

    if args.dataset == 'CIFAR10':
        trans_train, trans_test = build_trans_CIFAR(args.dataset)
        testset_clean = datasets.CIFAR10(args.data_dir, train=is_train, download=True, transform=trans_test)
        testset_poisoned = CIFAR10Poison(args, args.data_dir, train=is_train, download=True, transform=trans_test)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        transform = build_trans_MNIST(args.dataset)
        print("Transform = ", transform)
        testset_clean = datasets.MNIST(args.data_dir, train=is_train, download=True, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_dir, train=is_train, download=True, transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned

def build_trans_MNIST(dataset):
    transform = transforms.Compose([
            transforms.transforms.Resize([32, 32]),
            transforms.ToTensor(),
        ])
    return transform

def build_trans_CIFAR(dataset):
    if dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
   
    return transform_train, transform_test


# def build_transform(dataset):
    
    # if dataset == "CIFAR10":
    #     mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    # elif dataset == "MNIST":
    #     mean, std = (0.5,), (0.5,)
    # else:
    #     raise NotImplementedError()

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    #     ])
    # mean = torch.as_tensor(mean)
    # std = torch.as_tensor(std)
    # detransform = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()) # you can use detransform to recover the image
    
    # transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         ])
    # return transform, detransform
    # if dataset == "MNIST":
    #     transform = transforms.Compose([
    #         transforms.transforms.Resize([32, 32]),
    #         transforms.ToTensor(),
    #     ])
    # else:
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         ])
    
    # return transform