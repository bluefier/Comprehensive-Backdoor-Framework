import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from spikingjelly.datasets import play_frame
import os
from cgi import test
from doctest import ELLIPSIS_MARKER
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets import split_to_train_test_set

class PoisonedDataset(Dataset):

    def __init__(self, dataset, trigger_label=0, mode='train', epsilon=0.01, pos='top-left', trigger_type=0, time_step=16,
                 trigger_size=0.1, device=torch.device('cuda'), dataname='minst', polarity=0, data_dir='./data/'):

        # Handle special case for CIFAR10
        if type(dataset) == torch.utils.data.Subset:
            path = os.path.join(data_dir,  'cifar10split.pt')
            print("Path: ", path)

            if not os.path.exists(path):
                print("Targets")
                path_target = os.path.join(data_dir, 'targets.pt')
                if not os.path.exists(path_target):
                    targets = torch.Tensor(dataset.dataset.targets)[
                        dataset.indices]
                    torch.save(targets, path_target)
                else:
                    targets = torch.load(path_target)

                print("Data")
                path_data = os.path.join(data_dir, 'data_1.pt')
                if not os.path.exists(path_data):
                    data = np.array([i[0] for i in dataset.dataset])
                    torch.save(data, path_data)
                else:
                    data = torch.load(path_data)

                path_data = os.path.join(data_dir, 'data_2.pt')
                if not os.path.exists(path_data):
                    data = torch.Tensor(data)[dataset.indices]
                    torch.save(data, path_data)
                else:
                    data = torch.load(path_data)

                torch.save({'data': data, 'targets': targets}, path)
            else:
                data = torch.load(path)['data']
                targets = torch.load(path)['targets']

            dataset = dataset.dataset
        else:
            targets = dataset.targets
            data = dataset

        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

        self.time_step = time_step
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.transform = dataset.transform

        # TODO: Change the attributes of the imagenet to fit the same as MNIST
        self.data, self.targets = self.add_trigger(
            data, targets, trigger_label, epsilon, mode, pos, trigger_type, trigger_size, polarity)
        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):

        img = self.data[item]
        label_idx = int(self.targets[item])

        if self.transform:
            img = self.transform(img)

        label = np.zeros(self.class_num)
        label[label_idx] = 1  # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
      
        return self.data.shape[2:]

    def norm(self, data):
        offset = np.mean(data, 0)
        scale = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    def add_trigger(self, data, targets, trigger_label, epsilon, mode, pos, type, trigger_size, polarity):

        print("[!] Generating " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)

        # Fixes some bugs
        # if not isinstance(torch.Tensor, type(new_targets)):
        if not torch.is_tensor(new_targets):
            new_targets = torch.Tensor(new_targets)

        # Choose a random subset of samples to be poisoned
        perm = np.random.permutation(len(new_data))[
            0: int(len(new_data) * epsilon)]

        # NMIST/DVSGesture
        print(new_data)
        frame, label = new_data[0]
        #CIFAR10-DVS
        # # print(new_data.shape)
        # # print(new_data.type())
        # frame = new_data[0]
        # # print(frame.shape)
        width, height = frame.shape[2:]

        # Swap every samples to the target class
        new_targets[perm] = trigger_label

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

        #>>>>>>>IMPORTANT<<<<<<<<<<<<<<<
        #下面的代码在使用DVSGesture和NMNIST数据集的时候使用
        #CIFAR10-DVS的时候要注释掉
        new_data = np.array([i[0] for i in new_data])
        print('trans:', new_data.shape)
        list = []
        for data in tqdm(new_data):
            # data = np.array(data)
            # print('data shape:', data.shape)
            list.append(np.array(data))
        new_data = np.array(list)
        print('new_data shape:', new_data.shape)
        #>>>>>>>>>>>END<<<<<<<<<<<<<<<<<<<


        #>>>>>>>>>>>>>>IMPORTANT<<<<<<<<<<<<<<
        #下面这段代码中的newdata在DVSGesture和NMIST数据集的情况下是五维的【perm，T, channel，W, H】
        #在CIFAR10-DVS的情况下是五维的【perm，：，channel，W, H】
        # Static trigger
        if type == 0:
            # TODO: Take into account the polarity. Being 0 green, 1 ligth blue and 2 a mix of both ie dark blue
            # Check this im not sure

            #CIFAR10-DVS
            if polarity == 0:
                new_data[perm, :, 0, y_begin:y_end, x_begin:x_end] = 1
                new_data[perm, :, 1, y_begin:y_end, x_begin:x_end] = 0
            elif polarity == 1:
                new_data[perm, :, 0, y_begin:y_end, x_begin:x_end] = 0
                new_data[perm, :, 1, y_begin:y_end, x_begin:x_end] = 1
            else:
                new_data[perm, :, 0, y_begin:y_end, x_begin:x_end] = 1
                new_data[perm, :, 1, y_begin:y_end, x_begin:x_end] = 1

        else:
            # Dynamic trigger
            new_data = create_dynamic_trigger(
                size_width, size_height, new_data, height, width, perm, pos, self.time_step, polarity)
        
        print(
            f'Injecting Over: Bad Imgs: {len(perm)}. Clean Imgs: {len(new_data)-len(perm)}. Epsilon: {epsilon}')
        

        return torch.Tensor(new_data), new_targets

class PoisonedTestDataset(Dataset):

    def __init__(self, dataset, trigger_label=0, mode='train', epsilon=0.01, pos='top-left', trigger_type=0, time_step=16,
                 trigger_size=0.1, device=torch.device('cuda'), dataname='minst', polarity=0, data_dir='./data/'):

        # Handle special case for CIFAR10
        if type(dataset) == torch.utils.data.Subset:
            path = os.path.join(data_dir,  'cifar10splittest.pt')
            print("Path: ", path)

            if not os.path.exists(path):
                print("Targets")
                path_target = os.path.join(data_dir, 'targets_test.pt')
                if not os.path.exists(path_target):
                    targets = torch.Tensor(dataset.dataset.targets)[
                        dataset.indices]
                    torch.save(targets, path_target)
                else:
                    targets = torch.load(path_target)

                print("Data")
                path_data = os.path.join(data_dir, 'data_1_test.pt')
                if not os.path.exists(path_data):
                    data = np.array([i[0] for i in dataset.dataset])
                    torch.save(data, path_data)
                else:
                    data = torch.load(path_data)

                path_data = os.path.join(data_dir, 'data_2_test.pt')
                if not os.path.exists(path_data):
                    data = torch.Tensor(data)[dataset.indices]
                    torch.save(data, path_data)
                else:
                    data = torch.load(path_data)

                torch.save({'data': data, 'targets': targets}, path)

            else:
                data = torch.load(path)['data']    # 选取测试数据集
                targets = torch.load(path)['targets']

            dataset = dataset.dataset
        else:
            targets = dataset.targets
            data = dataset

        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

        self.time_step = time_step
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.transform = dataset.transform

        # TODO: Change the attributes of the imagenet to fit the same as MNIST
        self.data, self.targets = self.add_trigger(
            data, targets, trigger_label, mode, pos, trigger_type, trigger_size, polarity)
        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):

        img = self.data[item]
        label_idx = int(self.targets[item])

        if self.transform:
            img = self.transform(img)

        label = np.zeros(self.class_num)
        label[label_idx] = 1  # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):

        return self.data.shape[2:]

    def norm(self, data):
        offset = np.mean(data, 0)
        scale = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    def add_trigger(self, data, targets, trigger_label, mode, pos, type, trigger_size, polarity):

        if mode == 'test_clean':
            print("[!] Generating " + mode + " Imgs")
            new_data = copy.deepcopy(data)
            new_targets = copy.deepcopy(targets)

            new_data = np.array([i[0] for i in new_data])
            print('trans_before: ', new_data.shape)
            list = []
            for data in tqdm(new_data):
                list.append(np.array(data))

            new_data = np.array(list)
            print('trans_after: ', new_data.shape)

            print(
                f'Injecting Over: Clean Test Imgs: {len(new_data)-len(new_data)}. Clean Imgs: {len(new_data)}.')
        else:
            print("[!] Generating " + mode + " Bad Imgs")
            new_data = copy.deepcopy(data)
            new_targets = copy.deepcopy(targets)

            # Fixes some bugs
            # if not isinstance(torch.Tensor, type(new_targets)):
            if not torch.is_tensor(new_targets):
                new_targets = torch.Tensor(new_targets)

            # Choose a random subset of samples to be poisoned
            perm = np.random.permutation(len(new_data))[
                0: int(len(new_data))]

            # NMIST/DVSGesture
            print(new_data)
            frame, label = new_data[0]
            #CIFAR10-DVS
            # print(new_data.shape)
            # print(new_data.type())
            # frame = new_data[0]
            # # print(frame.shape)

            width, height = frame.shape[2:]

            # Swap every samples to the target class
            new_targets[perm] = trigger_label

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

            #>>>>>>>IMPORTANT<<<<<<<<<<<<<<<
            #下面的代码在使用DVSGesture和NMNIST数据集的时候使用
            #CIFAR10-DVS的时候要注释掉
            new_data = np.array([i[0] for i in new_data])
            print('trans:', new_data.shape)
            list = []
            for data in tqdm(new_data):
                list.append(np.array(data))
            new_data = np.array(list)
            print('new_data shape:', new_data.shape)
            #>>>>>>>>>>>END<<<<<<<<<<<<<<<<<<<


            #>>>>>>>>>>>>>>IMPORTANT<<<<<<<<<<<<<<
            #下面这段代码中的newdata在DVSGesture和NMIST数据集的情况下是四维的【perm，T, channel，W, H】
            #在CIFAR10-DVS的情况下是五维的【perm，：，channel，W, H】
            # Static trigger
            if type == 0:
                # TODO: Take into account the polarity. Being 0 green, 1 ligth blue and 2 a mix of both ie dark blue
                # Check this im not sure

                if polarity == 0:
                    new_data[perm, :, 0, y_begin:y_end, x_begin:x_end] = 1
                    new_data[perm, :, 1, y_begin:y_end, x_begin:x_end] = 0
                elif polarity == 1:
                    new_data[perm, :, 0, y_begin:y_end, x_begin:x_end] = 0
                    new_data[perm, :, 1, y_begin:y_end, x_begin:x_end] = 1
                else:
                    new_data[perm, :, 0, y_begin:y_end, x_begin:x_end] = 1
                    new_data[perm, :, 1, y_begin:y_end, x_begin:x_end] = 1

            else:
                # Dynamic trigger
                new_data = create_dynamic_trigger(
                    size_width, size_height, new_data, height, width, perm, pos, self.time_step, polarity)
            
            print(
                f'Injecting Over: Bad Imgs: {len(perm)}. Clean Imgs: {len(new_data)-len(perm)}.')
        
        return torch.Tensor(new_data), new_targets


def get_dataset(dataname, frames_number, data_dir):
    '''
    For a given dataname, return the train and testset

    Parameters:
        dataname (str): name of the dataset

    Returns:
        trainset (torch.utils.data.Dataset): train dataset
        testset (torch.utils.data.Dataset): test dataset
    '''

    '''
    Split_by splits the event to integrate them to frames. This can be done either by setting some fixed time or the number of frames.
    We choose the number of frames following the paper: https://arxiv.org/abs/2007.05785?context=cs.LG

    However the data_type 
    '''

    # data_dir = os.path.join(data_dir, dataname)

    if dataname == 'gesture':
        transform = None

        train_set = DVS128Gesture(
            data_dir, train=True, data_type='frame', split_by='number', frames_number=frames_number, transform=transform)
        test_set = DVS128Gesture(data_dir, train=False,
                                 data_type='frame', split_by='number', frames_number=frames_number, transform=transform)

    elif dataname == 'cifar10':

        # Split by number as in: https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

        dataset = CIFAR10DVS(data_dir, data_type='frame',
                             split_by='number', frames_number=frames_number)

        cifar = os.path.join(data_dir, 'cifar10.pt')
        if not os.path.exists(cifar):
            # TODO: Since this is slow, consider saving the dataset
            train_set, test_set = split_to_train_test_set(
                origin_dataset=dataset, train_ratio=0.9, num_classes=10)
            torch.save({'train': train_set, 'test': test_set}, cifar)

        else:
            data = torch.load(cifar)
            train_set = data['train']
            test_set = data['test']

    elif dataname == 'mnist':

        train_set = NMNIST(data_dir, train=True, data_type='frame',
                           split_by='number', frames_number=frames_number)

        test_set = NMNIST(data_dir, train=False, data_type='frame',
                          split_by='number', frames_number=frames_number)
    else:
        raise ValueError(f'{dataname} is not supported')
    
    print(f'Trainset: {len(train_set)}, Testset: {len(test_set)}')

    return train_set, test_set

def create_backdoor_data_loader(dataname, trigger_label, epsilon, pos, type, trigger_size,
                                batch_size_train, batch_size_test, T, device, data_dir, polarity):

    # Get the dataset
    train_data, test_data = get_dataset(dataname, T, data_dir)
    # print(train_data[0])
    # output_frame, label = train_data[0]
    # print('=====tain data before=====: ', output_frame.shape)
    # play_frame(output_frame)

    train_data = PoisonedDataset(
        train_data, trigger_label, mode='train', epsilon=epsilon, device=device,
        pos=pos, trigger_type=type, time_step=T, trigger_size=trigger_size, dataname=dataname, polarity=polarity, data_dir=data_dir)
    # frame_clean_train, label = train_data[56]
    # print('frame_clean_train: ', frame_clean_train.shape)

    test_data_ori = PoisonedTestDataset(test_data, trigger_label, mode='test_clean', epsilon=epsilon, device=device,
        pos=pos, trigger_type=type, time_step=T, trigger_size=trigger_size, dataname=dataname, polarity=polarity, data_dir=data_dir)
    # frame_clean_test, label = test_data_ori[56]
    # print('frame_clean_train: ', frame_clean_test.shape)

    test_data_tri = PoisonedTestDataset(test_data, trigger_label, mode='test_poisoned', epsilon=epsilon, device=device,
        pos=pos, trigger_type=type, time_step=T, trigger_size=trigger_size, dataname=dataname, polarity=polarity, data_dir=data_dir)

    frame_clean_train, label = train_data[56]
    play_frame(frame_clean_train, '/home/jinlingxin/SNN/RGA/trigger_pattern/clean_train_%s_%s_%s_%s.gif' % (polarity, dataname, trigger_label, epsilon))
    frame_clean_test, label = test_data_ori[56]
    play_frame(frame_clean_test, '/home/jinlingxin/SNN/RGA/trigger_pattern/clean_%s_%s_%s_%s.gif' % (polarity, dataname, trigger_label, epsilon))
    frame, label = test_data_tri[56]
    play_frame(frame, '/home/jinlingxin/SNN/RGA/trigger_pattern/trigger_%s_%s_%s_%s.gif' % (polarity, dataname, trigger_label, epsilon))

    train_data_loader = DataLoader(
        dataset=train_data,    batch_size=batch_size_train, shuffle=True)
    test_data_ori_loader = DataLoader(
        dataset=test_data_ori, batch_size=batch_size_test, shuffle=True)
    test_data_tri_loader = DataLoader(
        dataset=test_data_tri, batch_size=batch_size_test, shuffle=True)

    return train_data_loader, test_data_ori_loader, test_data_tri_loader

def create_dynamic_trigger(size_x, size_y, new_data, height, width, perm, pos, time_step, polarity):

    if pos == 'top-left':
        start_x = size_x + 2
        start_y = size_y + 2

        width_list = [start_x, start_x +
                      size_x + 2, start_x + size_x * 2 + 2]
        height_list = [start_y, start_y, start_y]
    elif pos == 'top-right':
        start_x = width - 2
        start_y = size_y + 2

        width_list = [start_x, start_x -
                      size_x - 2, start_x - size_x * 2 - 2]
        height_list = [start_y, start_y, start_y]
    elif pos == 'bottom-left':
        start_x = size_x + 2
        start_y = height - 2

        width_list = [start_x, start_x +
                      size_x + 2, start_x + size_x * 2 + 2]
        height_list = [start_y, start_y, start_y]
    elif pos == 'bottom-right':
        start_x = height - 2
        start_y = width - 2

        width_list = [start_x, start_x -
                      size_x - 2, start_x - size_x * 2 - 2]
        height_list = [start_y, start_y, start_y]
    elif pos == 'middle':
        start_x = int(width/2) - 2
        start_y = int(height/2) - 2

        width_list = [start_x, start_x +
                      size_x + 2, start_x + size_x * 2 + 2]
        height_list = [start_y, start_y, start_y]
    elif pos == 'random':
        start_x = np.random.randint(0, width)
        start_y = np.random.randint(0, height)

        width_list = [start_x, start_x +
                      size_x + 2, start_x + size_x * 2 + 2]
        height_list = [start_y, start_y, start_y]

    j = 0
    t = 0

    while t < time_step - 1:
        if j >= len(width_list):
            j = 0

        for x in range(size_x):
            for y in range(size_y):

                if polarity == 0:
                    new_data[perm, t, 0, height_list[j]-y, width_list[j]-x] = 1
                    new_data[perm, t + 1, 1, height_list[j] -
                             y, width_list[j]-x] = 0
                elif polarity == 1:
                    new_data[perm, t, 0, height_list[j]-y, width_list[j]-x] = 0
                    new_data[perm, t + 1, 1, height_list[j] -
                             y, width_list[j]-x] = 1
                else:
                    new_data[perm, t, 0, height_list[j]-y, width_list[j]-x] = 1
                    new_data[perm, t + 1, 1, height_list[j] -
                             y, width_list[j]-x] = 1

        j += 1
        t += 1

    return new_data