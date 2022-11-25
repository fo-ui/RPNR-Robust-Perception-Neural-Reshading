import torch.nn as nn
import torch

class AlexNet(nn.Module):

    def __init__(self, orig_model, num_classes = 15, device = 'cpu'):
        
        super(AlexNet, self).__init__()
        #self.b0 = nn.BatchNorm2d(6400).to(device)
        #self.b1 = nn.BatchNorm2d(4096)
        #self.b2 = nn.BatchNorm2d(4096)
        self.device = device
        self.orig = nn.Sequential(*(list(orig_model.children())[:-1])).to(device)
        for param in self.orig.parameters():
            param.requires_grad = True
        self.c1 = nn.Linear(256 * 6 * 6, 4096) #0
        self.c2 =  nn.ReLU(inplace=True)  # 1
        self.c3 = nn.Dropout(0.5)  # 2
        self.c4 = nn.Linear(4096, 4096)  # 3
        self.c5 = nn.ReLU(inplace=True)  # 4
        self.c6 = nn.Dropout(0.5)  # 5
        self.c7 = nn.Linear(4096, num_classes)  # 6
    

    def forward(self, x):
        x = self.orig(x)
        #x = self.b0(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x6 = self.c6(x5)
        x = self.c7(x6)
        return x, x1, x4