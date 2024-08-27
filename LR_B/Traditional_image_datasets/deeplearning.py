import torch
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torch.cuda import amp
from spikingjelly.activation_based import encoding, functional

encoder = encoding.PoissonEncoder()
# F.mse_loss = F.mse_loss()

def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = torch.optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = torch.optim.Adam(param, lr=lr)
    return optimizer

def train_one_epoch(data_loader, model, optimizer, loss_mode, device, scaler, T, start_time):
   
    model.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    for img, label in data_loader:
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        label_onehot = F.one_hot(label, 10).float()

        if scaler is not None:
            with amp.autocast():
                out_fr = 0.
                for t in range(T):
                    encoded_img = encoder(img)  #输入的是图像的泊松编码
                    out_fr += model(encoded_img)
                out_fr = out_fr / T
                loss = F.mse_loss(out_fr, label_onehot)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_fr = 0.
            for t in range(T):
                encoded_img = encoder(img)
                out_fr += model(encoded_img)
            out_fr = out_fr / T
            loss = F.mse_loss(out_fr, label_onehot)
            loss.backward()
            optimizer.step()

        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(model)
    
    train_time = time.time()
    train_speed = train_samples / (train_time - start_time)
    train_loss /= train_samples
    train_acc /= train_samples

    return {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_speed": train_speed,
            "train_time": train_time,
            }

def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device, T, train_time):

    ta = eval(data_loader_val_clean, model, device, T, train_time, print_perform=True)
    asr = eval(data_loader_val_poisoned, model, device, T, train_time, print_perform=False)
    
    return {
            'clean_acc': ta['acc'], 
            'clean_loss': ta['loss'],
            'asr': asr['acc'], 
            'asr_loss': asr['loss'],
            }

def eval(data_loader, model, device, T, train_time, print_perform=False):
    
    model.eval() # switch to eval status
    test_loss = 0
    test_acc = 0
    test_samples = 0

    for img, label in tqdm(data_loader):

        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        label_onehot = F.one_hot(label, 10).float()
        out_fr = 0.
        for t in range(T):
            encoded_img = encoder(img)  #输入的是图像的泊松编码
            # print('test shape: ', encoded_img.shape)
            out_fr += model(encoded_img)
        out_fr = out_fr / T
        loss = F.mse_loss(out_fr, label_onehot)

        test_samples += label.numel()
        test_loss += loss.item() * label.numel()
        test_acc += (out_fr.argmax(1) == label).float().sum().item()
        functional.reset_net(model)

    test_time = time.time()
    test_speed = test_samples / (test_time - train_time)
    test_loss /= test_samples
    test_acc /= test_samples

    # if print_perform:
    #     print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return {
            "acc": test_acc,
            "loss": test_loss,
            "speed": test_speed,
            "time": test_time,
            }