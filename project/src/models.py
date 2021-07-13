import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class Lowlevel_features(nn.Module):
    def __init__(self):
        super(Lowlevel_features, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1,out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_layer3 = nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_layer5 = nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv_layer6 = nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = F.relu(self.conv_layer1(x))
        y = F.relu(self.conv_layer2(y))
        y = F.relu(self.conv_layer3(y))
        y = F.relu(self.conv_layer4(y))
        y = F.relu(self.conv_layer5(y))
        y = F.relu(self.conv_layer6(y))
        return y

class Midlevel_features(nn.Module):
    def __init__(self):
        super(Midlevel_features, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=512,out_channels=256, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        y = F.relu(self.conv_layer1(x))
        y = F.relu(self.conv_layer2(y))
        return y

class Global_features(nn.Module):
    def __init__(self):
        super(Global_features, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_layer3 = nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, stride=1, padding=1)
        self.full_layer5 = nn.Linear(in_features=7*7*512, out_features=1024)
        self.full_layer6 = nn.Linear(in_features=1024, out_features=512)
        self.full_layer7 = nn.Linear(in_features=512, out_features=256)
    
    def forward(self, x):
        y = F.relu(self.conv_layer1(x))
        y = F.relu(self.conv_layer2(y))
        y = F.relu(self.conv_layer3(y))
        y = F.relu(self.conv_layer4(y)).view(-1,7*7*512)
        y = F.relu(self.full_layer5(y))
        y = F.relu(self.full_layer6(y))
        y = F.relu(self.full_layer7(y))
        return y

class Colorization(nn.Module):
    def __init__(self):
        super(Colorization, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=256,out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=128,out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_layer3 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64,out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_layer5 = nn.Conv2d(in_channels=32,out_channels=2, kernel_size=3, stride=1, padding=1)
        
    def upsample(self, x, scale=2):
        return F.interpolate(input=x, scale_factor=2)

    def forward(self, x):
        y = F.relu(self.conv_layer1(x))
        y = self.upsample(y)
        y = F.relu(self.conv_layer2(y))
        y = F.relu(self.conv_layer3(y))
        y = self.upsample(y)
        y = F.relu(self.conv_layer4(y))
        y = F.sigmoid(self.conv_layer5(y))
        y = self.upsample(y)
        return y


class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer,self).__init__()
        self.lowLevelFeaturesNet = Lowlevel_features()
        self.midLevelFeaturesNet = Midlevel_features()
        self.globalFeaturesNet = Global_features()
        self.colorizationNet = Colorization()
        self.fusion_ = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
    
    def fusion_layer(self, midlevel_features, global_features):
        global_features_stacked = torch.stack([torch.stack([global_features for i in range(28)]) for j in range(28)]).permute(2,3,0,1)
        fused = torch.cat((midlevel_features, global_features_stacked), dim=1)
        fused = F.relu(self.fusion_(fused))
        return fused

    def forward(self, gray_scale_img):
        gray_scale_img = gray_scale_img.view(-1,1,224,224).float()
        lowlevel_features = self.lowLevelFeaturesNet(gray_scale_img)

        # if gray_scale_img.shape != (224,224):
        #     lowlevel_features_scaled = self.lowLevelFeaturesNet(cv2.resize(gray_scale_img, (224,224)))
        # else:
        lowlevel_features_scaled = lowlevel_features

        midlevel_features = self.midLevelFeaturesNet(lowlevel_features)
        global_features = self.globalFeaturesNet(lowlevel_features_scaled)

        fused = self.fusion_layer(midlevel_features, global_features)

        output = self.colorizationNet(fused)
        return output.permute(0,2,3,1)