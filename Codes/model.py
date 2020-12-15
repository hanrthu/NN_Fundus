import torchvision
import torch.nn as nn
import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
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

class siamese_senet(nn.Module):
    def __init__(self,classnum=2, name="siamese_senet18", pretrain=True,siamese='concatenation'):
        super(siamese_senet, self).__init__()
        self.siamese=siamese
        self.name = name
        print("Pretrain:",pretrain)
        if name == "siamese_senet18":
            self.model = se_resnet_18(pretrained=pretrain)
        elif name == "siamese_senet50":
            self.model = se_resnet_50(pretrained=pretrain)
        elif name == "siamese_resnet18":
            self.model = torchvision.models.resnet18(pretrained=pretrain)
        elif name == "siamese_resnet50":
            self.model = torchvision.models.resnet50(pretrained=pretrain)
        else:
            assert False
        if self.siamese == 'concatenation':
            print("using concatenation")
            channel_in = 4*self.model.fc.in_features
            if "siamese_resnet" in name:
                self.model.fc = nn.Identity()
                self.fc = nn.Linear(in_features = channel_in, out_features = classnum)
            else:
                self.model.fc = nn.Linear(in_features = channel_in, out_features = classnum)
        elif self.siamese == 'projection':
            print("using projection")
            channel_in = self.model.fc.in_features
            self.W = Parameter(torch.zeros((channel_in,channel_in)))
            width = 32
            init.xavier_normal_(self.W)
            self.model.fc = nn.Linear(in_features = width, out_features = classnum)
    def forward(self,X):
        assert X.shape[1] == 6
        torch.cuda.empty_cache()
        part1 = X[:,:3,:,:]
        part2 = X[:,3:,:,:]
        if "siamese_resnet" in self.name:
            feature1 = self.model(part1)
            feature2 = self.model(part2)
            if self.siamese=='concatenation':
                torch.cuda.empty_cache()
                feature = torch.cat((feature1,feature2,feature1-feature2,feature1*feature2),-1)
                output = self.fc(feature)
                return output
            elif self.siamese=='projection':
                interaction = torch.matmul(torch.matmul(feature1, self.W), (torch.matmul(feature2, self.W)).T)
                pred = torch.diagonal(interaction,0)
                pred = pred/torch.max(pred)
                output = torch.stack((pred,1-pred),1)
                return output
        else:
            feature1 = self.model.get_features(part1)
            feature2 = self.model.get_features(part2)
            if self.siamese=='concatenation':
                torch.cuda.empty_cache()
                feature = torch.cat((feature1,feature2,feature1-feature2,feature1*feature2),-1)
                output = self.model.fc(feature)
                return output
            elif self.siamese=='projection':
                interaction = torch.matmul(torch.matmul(feature1, self.W), (torch.matmul(feature2, self.W)).T)
                pred = torch.diagonal(interaction,0)
                pred = pred/torch.max(pred)
                output = torch.stack((pred,1-pred),1)
                return output
