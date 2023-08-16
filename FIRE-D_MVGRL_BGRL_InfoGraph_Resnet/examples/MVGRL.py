import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import os
from torch import nn
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import numpy as np
import torch
from evaluate_embedding import evaluate_embedding
from torch_geometric.data import Dataset, Data


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index)
            z = self.activation(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, g


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, gcn1, gcn2, mlp1, mlp2, aug1, aug2):
        super(Encoder, self).__init__()
        self.gcn1 = gcn1
        self.gcn2 = gcn2
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.aug1 = aug1
        self.aug2 = aug2

    def forward(self, x, edge_index, batch):
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index)
        z1, g1 = self.gcn1(x1, edge_index1, batch)
        z2, g2 = self.gcn2(x2, edge_index2, batch)
        h1, h2 = [self.mlp1(h) for h in [z1, z2]]
        g1, g2 = [self.mlp2(g) for g in [g1, g2]]
        return h1, h2, g1, g2


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    x = []
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        h1, h2, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g1)
        #print(h1.size(),h2.size(),g1.size(),g2.size())
        loss = contrast_model(h1=h1, h2=h2, g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    x = torch.cat(x, dim=0)
    #print(x.size())
    return epoch_loss,x


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g1 + g2)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    #split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    #result = SVMEvaluator(linear=True)(x, y, split)
    return x


def main():
    device = torch.device('cuda')
       
    file_list = os.listdir('data/')
    data_list = [torch.load('data/'+file_list[i]) for i in range(len(file_list))]
    for data in data_list:
        data['x'] = data['x'].to(torch.float32)
        #print(data['x'])        
    dataloader = DataLoader(data_list, batch_size=5)
    input_dim = 5
    
   
    ans = [0]*10
    for i in range(10):
        torch.manual_seed(i)
        aug1 = A.Identity()
        aug2 = A.PPRDiffusion(alpha=0.2, use_cache=False)
        gcn1 = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
        gcn2 = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
        mlp1 = FC(input_dim=32, output_dim=32)
        mlp2 = FC(input_dim=32 * 2, output_dim=32)
        encoder_model = Encoder(gcn1=gcn1, gcn2=gcn2, mlp1=mlp1, mlp2=mlp2, aug1=aug1, aug2=aug2).to(device)
        contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)
    
        optimizer = Adam(encoder_model.parameters(), lr=0.01)
        
        with tqdm(total=1, desc='(T)') as pbar:
            for epoch in range(1, 11):
                loss,g = train(encoder_model, contrast_model, dataloader, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()
                
	torch.save(g,'NASA_EMBEDDING_InfoGraph.pt')           
        labels = np.loadtxt('labels_NASA.txt')
        labels =torch.tensor( labels)
        labels =np.array(labels.detach().cpu().numpy())
        
        emb = np.array(torch.load('NASA_EMBEDDING_MVGRL.pt').detach().cpu().numpy())
        ans[i] = evaluate_embedding(emb,labels)[0]
    np.mean(ans)
    np.std(ans)
if __name__ == '__main__':
    main()



