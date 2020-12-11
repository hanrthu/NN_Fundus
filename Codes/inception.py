from torchvision import models
import torch.nn as nn
import torch

class inception_v3(nn.Module):
    def __init__(self):
        super(inception_v3,self).__init__()
        self.model = models.inception_v3(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.AuxLogits.fc = nn.Linear(768,2)
        self.model.fc = nn.Linear(2048,2)
        self.model.aux_logits=False

        self.loss = nn.CrossEntropyLoss()
    def forward(self,x,y):
        results = self.model(x)
        loss = self.loss(results,y)
        pred = torch.argmax(logits, 1)
        correct_pred = (pred.int() == y.int())
        acc = torch.mean(correct_pred.float())
        return loss,acc