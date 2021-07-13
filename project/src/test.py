import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import models

model = models.Colorizer()
model = torch.load('best_tr_model1')

in_path = sys.argv[1]
out_path = sys.argv[2]

L = torch.tensor(cv2.resize(cv2.imread(in_path,0),(224,224))/255)
L = torch.unsqueeze(L,0)

ab = model(L)

L = torch.unsqueeze(L,-1)

Lab = torch.squeeze(torch.cat((L,ab), dim=-1)).detach().numpy().astype('float32')

orig = cv2.imread('lab.png')

Lab[:,:,0]*=255
Lab[:,:,1:]*=254.0

Lab = Lab.astype('uint8')

RGB = cv2.cvtColor(Lab, cv2.COLOR_LAB2RGB)
cv2.imwrite(out_path, RGB)
