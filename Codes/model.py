import torchvision
import torch.nn as nn
import torch
from senet import se_resnet_18, se_resnet_50

class resnet(nn.Module):
    def __init__(self, classnum=2, name="resnet18", pretrain=True, stack=False):
        super(resnet, self).__init__()
        if name == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=pretrain)
        elif name == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=pretrain)
        else:
            assert False    
        channel_in = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features = channel_in, out_features = classnum)
        if stack:
            self.model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, X):
        return self.model(X)

class inception(nn.Module):
    def __init__(self, classnum=2, pretrain=True):
        super(inception, self).__init__()
        self.model = torchvision.models.inception_v3(pretrained=pretrain)

        self.model.AuxLogits.fc = nn.Linear(768, classnum)
        self.model.fc = nn.Linear(2048, classnum)
        self.model.aux_logits = False
    
    def forward(self, X):
        return self.model(X)

class SEnet(nn.Module):
    def __init__(self, classnum=2, name="senet50", pretrain=True, stack=False):
        super(SEnet, self).__init__()
        if name == "senet18":
            self.model = se_resnet_18(pretrained=pretrain)
        elif name == "senet50":
            self.model = se_resnet_50(pretrained=pretrain)
        else:
            assert False    
        channel_in = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features = channel_in, out_features = classnum)
        if stack:
            self.model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, X):
        return self.model(X)    