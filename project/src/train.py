import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data.dataset import random_split

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import models
from tqdm import tqdm

dataset_path = './Dataset/'
saved_models_path = './saved_models'

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.L = X
        self.ab = Y
 
    def __len__(self):
        return len(self.L)
 
    def __getitem__(self, index):
        return (self.L[index]/100), (self.ab[index]/100)

class DeviceLoader():
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for batch in self.loader:
            yield to_device(batch, self.device)

# def train(train_dl, val_dl, epochs=2, batch_size=16):
def train(X_train, Y_train, X_val, Y_val, model, epochs=2):
    train_batches = DataLoader([{'gray': X_train[i]/255, 'ab': Y_train[i]/255} for i in range(len(X_train))], batch_size = 32)
    val_batches = DataLoader([{'gray': X_val[i]/255, 'ab': Y_val[i]/255} for i in range(len(X_val))], batch_size = 32)

    optimizer = optim.Adam(model.parameters())
    loss_fun = nn.MSELoss()

    best_tr_loss = float('inf')
    best_val_loss = float('inf')
    for epoch in range(50):
        epoch_loss = 0
        # total_samples = 0
        # for batch_no, batch in enumerate(tqdm(train_dl)):
        for batch_no, batch in enumerate(tqdm(train_batches)):
            optimizer.zero_grad()
            #print(batch['gray'].shape)
            y = model(batch['gray'])
            if batch_no == 0:
                aa = torch.cat((torch.unsqueeze(batch['gray'][0],-1),batch['ab'][0]), dim=-1).detach().numpy()
                bb = torch.cat((torch.unsqueeze(batch['gray'][0],-1),y[0]), dim=-1).detach().numpy()
                print(aa[0,0,:],bb[0,0,:])
                print(aa[106,143,:],bb[106,143,:])

                # print(aa.shape)
                # print(bb.shape)
                # cv2.imwrite('1b_original.png',aa)
                # cv2.imwrite('1b_original_rgb.png',cv2.cvtColor(aa, cv2.COLOR_LAB2RGB))
                # cv2.imwrite('1b_pred.png',bb)
                # cv2.imwrite('1b_pred_rgb.png',cv2.cvtColor(bb, cv2.COLOR_LAB2RGB))
            loss = loss_fun(batch['ab'], y)
            #print("Batch {} : Loss = {}".format(batch_no+1, loss.item()))
            epoch_loss+=loss.item()
            loss.backward()
            optimizer.step()
            # total_samples+=1
        epoch_loss/=len(train_batches)
        print("Epoch {} completed, Loss = {}".format(epoch, epoch_loss),flush=True)

        val_loss = 0
        # total_samples = 0
        with torch.no_grad():
            # for batch_no, batch in enumerate(tqdm(val_dl)):
            for batch_no, batch in enumerate(tqdm(val_batches)):
                y = model(batch['gray'])

                loss = loss_fun(batch['ab'], y)
                val_loss+=loss.item()
                # total_samples+=1
        val_loss/=len(val_batches) 

        print("Val Loss = {}".format(val_loss))
        if epoch_loss < best_tr_loss:
            torch.save(model, 'best_tr_model1')
        # if epoch_loss < best_val_loss:
        #     torch.save(model, 'best_val_model')


# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

# print("Device:", device)

print("Loading dataset....",flush=True)

X = np.load(dataset_path+'l/gray_scale.npy')
Y1 = np.load(dataset_path+'ab/ab1.npy')
Y2 = np.load(dataset_path+'ab/ab2.npy')
Y3 = np.load(dataset_path+'ab/ab3.npy')

print("Dataset loaded",flush=True)

Y = np.vstack((Y1,Y2,Y3))

# print(X.shape)
# print(np.max(X))1
# print(np.min(X))
# print(np.max(Y))
# print(np.min(Y))

# ds = Dataset(X, Y)

# tr_size = int(0.8*len(X))
# val_size = int(0.1*len(X))
# ts_size = int(0.1*len(X))

# tr_ds, val_ds, ts_ds = random_split(ds, [tr_size, val_size, ts_size])
# tr_dl = DeviceLoader(tr_ds, device)
# val_dl = DeviceLoader(val_ds, device)
# ts_dl = DeviceLoader(ts_ds, device)

X_train, X_test, Y_train, Y_test = train_test_split(X[:10000],Y[:10000],test_size = 0.2)
X_val, X_train, Y_val, Y_train = train_test_split(X_train, Y_train, test_size = 0.5)

print("Split finished...")
# model = to_device(models.Colorizer(), device)
model = models.Colorizer()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


model.apply(init_weights)

# cv2.imwrite('gray.png', X_train[0])
# print(X_train[0].shape, Y_train[].shape)
# cv2.imwrite('colour.png', np.concat((X_train[0], Y_train[0])))
train(X_train[:], Y_train[:], X_val[:], Y_val[:], model)
# train(tr_dl, val_dl, model)