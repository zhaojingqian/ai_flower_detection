import glob
import os
import time

# import timm

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import argparse
# 创建解析
parser = argparse.ArgumentParser(description="train flower classify",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 添加参数
parser.add_argument('--train_url', type=str,
                    help='the path model saved')
parser.add_argument('--data_url', type=str, help='the training data')
# 解析参数
args, unkown = parser.parse_known_args()

path = args.data_url
model_path = args.train_url
print(os.listdir(path))
# path = '/flower-discrimination/flower_photos/'
w = 100
h = 100
c = 3

def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)] 
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        print(idx, folder)
    for idx, folder in enumerate(cate):
        
        for im in glob.glob(folder + '/*.jpg'):
            img = cv2.imread(im)
            img = cv2.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

data, label = read_img(path)

data = torch.FloatTensor(data).permute(0 ,3, 1, 2)
label = torch.LongTensor(label)

seed = 109
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data = data / 255
(x_train, x_val, y_train, y_val) = train_test_split(data, label, test_size=0.20, random_state=seed)
# x_train = x_train / 255
# x_val = x_val / 255

flower_dict = {0:'bee', 1:'blackberry', 2:'blanket', 3:'bougainvilliea', 4:'bromelia', 5:'foxglove'}
class_list = ['bee', 'blackberry', 'blanket', 'bougainvilliea', 'bromelia', 'foxglove']

class FlowerDataset(Dataset):
    def __init__(self, data, label):
        super(FlowerDataset, self).__init__()
        self.data = data.to(device)
        self.label = label.to(device)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # self.attention = SelfAttention(12*12)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(12*12*64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 6),
        )

    def forward(self, data):
        x = self.conv_model(data)
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        return x
    
num_epochs = 200

class Stat:
    def __init__(self, training, writer=None):
        self.step = 0
        self.loss = []
        self.labels = []
        self.pred_labels = []
        self.training = training
        self.writer = writer
    
    def add(self, pred, labels, loss):
        labels = labels.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        pred_labels = np.argmax(pred, axis = 1)
        self.loss.append(loss)
        self.labels.extend(labels)
        self.pred_labels.extend(pred_labels)

    def log(self):
        self.step += 1
        acc = accuracy_score(self.labels, self.pred_labels)
        loss = sum(self.loss) / len(self.loss)
        self.loss = []
        self.labels = []
        self.pred_labels = []
        # if not self.writer:
        #     return loss, acc
        # if self.training:
        #     self.writer.add_scalar('train_loss', loss, self.step)
        #     self.writer.add_scalar('train_acc', acc, self.step)
        # else:
        #     self.writer.add_scalar('dev_loss', loss, self.step)
        #     self.writer.add_scalar('dev_acc', acc, self.step)
        return loss, acc
    
def train(model, train_data_loader, dev_data_loader):
    loss_func = CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.01,
    )

    # scheduler = timm.scheduler.CosineLRScheduler(
    #     optimizer=optimizer,
    #     t_initial=num_epochs,
    #     lr_min = 1e-5,
    #     warmup_t = 4,
    #     warmup_lr_init = 1e-4
    # )

    # writer = SummaryWriter('./summary/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    train_stat = Stat(training=True)#, writer=writer)
    dev_stat = Stat(training=False)#, writer=writer)


    best_acc, best_net = 0.0, None
    for epoch in range(num_epochs):
        print(f"--- epoch: {epoch + 1} ---")
        # scheduler.step(epoch)
        # lr = optimizer.param_groups[0]['lr']
        # print(f'lr = {lr}')
        for iter, batch in enumerate(train_data_loader):
            model.train()
            data, labels = batch[0], batch[1]
            pred_outputs = model(data)
            loss = loss_func(pred_outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_stat.add(pred_outputs, labels, loss.item())
            
        train_loss, train_acc = train_stat.log()

        model.eval()
        with torch.no_grad():
            for batch in dev_data_loader:
                data, labels = batch[0], batch[1]
                pred_outputs = model(data)
                loss = loss_func(pred_outputs, labels)
                dev_stat.add(pred_outputs, labels, loss.item())
        dev_loss, dev_acc = dev_stat.log()
        print(  f"training loss: {train_loss:.4f}, acc: {train_acc:.2%}, " \
                f"dev loss: {dev_loss:.4f}, acc: {dev_acc:.2%}.")

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_net = deepcopy(model)
            
    print(f"best dev acc: {best_acc:.4f}")
    return best_net


train_set = FlowerDataset(x_train, y_train)
dev_set = FlowerDataset(x_val, y_val)

train_data_loader = DataLoader(train_set, 32, True)
dev_data_loader = DataLoader(dev_set, 32, False)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

from sklearn.model_selection import StratifiedKFold

K = 5
skf = StratifiedKFold(n_splits=K, random_state=seed, shuffle=True)
models = []
for fold, (train_idx, val_idx) in enumerate(skf.split(data, label)):
    print(f'===========fold:{fold+1}==============')
    train_data, train_label = data[train_idx], label[train_idx]
    val_data, val_label = data[val_idx], label[val_idx]
    train_set = FlowerDataset(train_data, train_label)
    dev_set = FlowerDataset(val_data, val_label)

    train_data_loader = DataLoader(train_set, 32, True)
    dev_data_loader = DataLoader(dev_set, 32, False)

    net = MyModel().to(device)
    net.apply(weights_init)
    best_net = train(net, train_data_loader, dev_data_loader)
    torch.save(best_net.state_dict(), model_path + f"/fold{fold}-flower_mlp.pt")