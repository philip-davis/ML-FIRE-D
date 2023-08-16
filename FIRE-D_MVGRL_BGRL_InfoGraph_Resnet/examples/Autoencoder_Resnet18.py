#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:25:56 2023

@author: zhiwei
"""

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from torch.nn.functional import normalize
if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')
import numpy as np
from evaluate_embedding import evaluate_embedding




def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 1000
batch_size = 5
learning_rate = 0.00001

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])




i=0
file_list = os.listdir('embedding_resnet_18')
dataset =torch.load('embedding_resnet/'+file_list[0])
for i in range(1,3400):
    file = torch.load('embedding_resnet_18/'+file_list[i])
    dataset =torch.cat((dataset,file),0)
    
    

    
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(    # b, 1, 32, 1
        nn.ReLU(True),
            nn.Conv2d(1,16, (998,1), stride=(1,1), padding=(1,2)),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            
            nn.ConvTranspose2d(8, 1, (988, 1), stride=(1, 1), padding=(1, 7)),  # b, 1, 512, 1
            nn.Tanh()
        )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
            #print(x.size())
        for layer in self.decoder:
            x = layer(x)
            #print(x.size())
        #x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

from sklearn import preprocessing

ans = [0]*10
for i in range(10):
    torch.manual_seed(i)
    for epoch in range(500):
        total_loss = 0
        k=0
        for data in dataloader:
            #img, _ = data
            k+=1
            #print(k)
    
            data = normalize(data, p=2.0, dim = 1)
            img = data
            img= torch.reshape(img,(5,1,1000,1)) if k!=681 else torch.reshape(img,(4,1,1000,1))
            img = Variable(img).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, total_loss))
    ans[i] = total_loss


torch.save(model.state_dict(), './conv_autoencoder.pth')
