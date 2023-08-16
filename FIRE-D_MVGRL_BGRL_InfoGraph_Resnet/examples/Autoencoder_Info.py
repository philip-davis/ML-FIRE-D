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


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 1000
batch_size = 10
learning_rate = 0.0000001

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

#dataset = MNIST('./data_new', transform=img_transform, download=True)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataset = torch.load('NASA_EMBEDDING_BGRL.pt')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(    # b, 1, 32, 1
        nn.ReLU(True),
            nn.Conv2d(1,16, (32,1), stride=(1,1), padding=(2,2)),  # b, 16, 5, 5
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
            
            nn.ConvTranspose2d(8, 1, (20, 1), stride=(1, 1), padding=(1, 7)),  # b, 1, 512, 1
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

from sklearn import preprocessing


for i in range(10):
    torch.manual_seed(i)
    for epoch in range(300):
        total_loss = 0
        k=0
        for data in dataloader:
            #img, _ = data
            k+=1
            #print(k)
    
            data = normalize(data, p=2.0, dim = 1)
            img = data
            img= torch.reshape(img,(10,1,32,1)) if k!=681 else torch.reshape(img,(4,1,64,1))
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
        

torch.save(model.state_dict(), './conv_autoencoder.pth')



