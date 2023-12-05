## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

################################################################################################################
############### Facial Key Points Detection using Deep Convolutional Neural Network - NaimishNet ###############
################################################################################################################
class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.avgpool = nn.AdaptiveAvgPool2d((96, 96))
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.6)
        
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.avgpool(x)
        
        x = self.dropout1(self.maxpool1(F.elu(self.conv1(x))))
        x = self.dropout2(self.maxpool2(F.elu(self.conv2(x))))
        x = self.dropout3(self.maxpool3(F.elu(self.conv3(x))))
        x = self.dropout4(self.maxpool4(F.elu(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        # print("x : ", x.size())
        
        x = self.dropout5(F.elu(self.fc1(x)))
        x = self.dropout6(F.elu(self.fc2(x)))
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

##############################################################
############### NaimishNet with Residual Block ###############
##############################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, padding)

    def forward(self, x):
        residual = x.clone()
        out = self.batch_norm(self.conv1(x))
        out = self.relu(out)
        out = self.batch_norm(self.conv2(out))
        residual = self.batch_norm(self.conv1x1(residual)) # downsample
        out += residual  # Residual connection
        out = self.relu(out)
        return out

class NaimishResidualNet(nn.Module):

    def __init__(self):
        super(NaimishResidualNet, self).__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((96, 96))
        
        self.conv1 = ResidualBlock(1, 32, 4, 1, 3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = ResidualBlock(32, 64, 3, 1, 2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = ResidualBlock(64, 128, 2, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.3)
        
        self.conv4 = ResidualBlock(128, 256, 1, 1, 0)
        self.maxpool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(12544, 1000)
        self.dropout5 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1000, 1000)
        self.dropout6 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(1000, 136)

    def forward(self, x):
        x = self.avgpool(x)
        
        x = self.dropout1(self.maxpool1(self.conv1(x)))
        x = self.dropout2(self.maxpool2(self.conv2(x)))
        x = self.dropout3(self.maxpool3(self.conv3(x)))
        x = self.dropout4(self.maxpool4(self.conv4(x)))

        x = x.view(x.size(0), -1)
        # print("x : ", x.size())

        x = self.dropout5(F.elu(self.fc1(x)))
        x = self.dropout6(F.elu(self.fc2(x)))
        x = self.fc3(x)

        return x

class NaimishResidualNet_mini(nn.Module):

    def __init__(self):
        super(NaimishResidualNet_mini, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((96, 96))
        
        self.conv1 = ResidualBlock(1, 32, 4, 1, 3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = ResidualBlock(32, 64, 3, 1, 2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.2)
               
        self.fc1 = nn.Linear(46656, 1000)
        self.dropout3 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(1000, 1000)
        self.dropout4 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(1000, 136)

    def forward(self, x):
        x = self.avgpool(x)
        
        x = self.dropout1(self.maxpool1(self.conv1(x)))
        x = self.dropout2(self.maxpool2(self.conv2(x)))

        x = x.view(x.size(0), -1)
        # print("x : ", x.size())

        x = self.dropout3(F.elu(self.fc1(x)))
        x = self.dropout4(F.elu(self.fc2(x)))
        x = self.fc3(x)

        return x
        
######################################
############### ResNet ###############
######################################
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, block, layers):
        super().__init__()
        
        self.inplanes = 64
        
        self.avgpool = nn.AdaptiveAvgPool2d((96, 96))

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , 136)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.avgpool(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x