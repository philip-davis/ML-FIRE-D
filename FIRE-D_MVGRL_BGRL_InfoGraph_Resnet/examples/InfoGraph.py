import torch
import os.path as osp
import GCL.losses as L

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import os

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    x =[]
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        z, g = encoder_model(data.x, data.edge_index, data.batch)
        z, g = encoder_model.project(z, g)
        #print(g.size())
        loss = contrast_model(h=z, g=g, batch=data.batch)
        loss.backward()
        optimizer.step()
        x.append(g)
        epoch_loss += loss.item()
    x = torch.cat(x, dim=0)
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
        z, g = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


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
        gconv = GConv(input_dim=input_dim, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
        fc1 = FC(hidden_dim=32 * 2)
        fc2 = FC(hidden_dim=32 * 2)
        encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
        contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)
    
        optimizer = Adam(encoder_model.parameters(), lr=0.01)
    
        with tqdm(total=1000, desc='(T)') as pbar:
            for epoch in range(1, 11):
                loss,g = train(encoder_model, contrast_model, dataloader, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()
        sum(p.numel() for p in contrast_model.parameters())


        torch.save(g,'NASA_EMBEDDING_InfoGraph.pt')            
        #test_result = test(encoder_model, dataloader)
        #print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
        labels = np.loadtxt('labels_NASA.txt')   
        labels =torch.tensor( labels)
        labels =np.array(labels.detach().cpu().numpy())
        emb = np.array(torch.load('NASA_EMBEDDING_InfoGraph.pt').detach().cpu().numpy())
        ans[i] = evaluate_embedding(emb,labels)[0]
    np.mean(ans)
    np.std(ans)
if __name__ == '__main__':
    main()
