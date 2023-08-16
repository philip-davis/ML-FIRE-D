__author__ = 'SherlockLiao'

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
from sklearn import preprocessing
import numpy as np

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 1000
batch_size = 100
learning_rate = 0.0001

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

fileList = ["0-epoch50_16.npz",
            "1-epoch50_16.npz",
            "2-epoch50_16.npz",
            "3-epoch50_16.npz",
            "4-epoch50_16.npz",
            "5-epoch50_16.npz",
            "6-epoch50_16.npz",
            "7-epoch50_16.npz",
            "8-epoch50_16.npz",
            "9-epoch50_16.npz",
            ]



# dataset = torch.load('./result/epoch100_16_embed.pt')
fileName = fileList[5]
dataset = torch.FloatTensor(np.load("./result/"+fileName)['arr_0'])

print(dataset.size()) # ([3400,16])
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(1,16, (16,1), stride=(1,1), padding=(2,2)),  # b, 16, 5, 5
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
            nn.ConvTranspose2d(8, 1, (4, 1), stride=(1, 1), padding=(1, 7)),  # b, 1, 512, 1
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encoder(x)
        # print(x.size())
        x = self.decoder(x)
        return x


model = autoencoder()#.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    total_loss = 0
    k=0
    for data in dataloader:
        # img, _ = data
        k += 1
        data = normalize(data, p=2.0, dim = 1)

        img = data
        img = torch.reshape(img, (100, 1, 16, 1)) 
        img = Variable(img)#.cuda()
        img = img.float()
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
    # if epoch % 10 == 0:
    #     pic = to_img(output.cpu().data)
    #     save_image(pic, './dc_img/image_{}.png'.format(epoch))
print(img)
torch.save(model.state_dict(), './result/'+fileName+'_conv_autoencoder.pth')