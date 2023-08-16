import copy
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import SVMEvaluator, get_split
from GCL.models import BootstrapContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import os
import numpy as np
from evaluate_embedding import evaluate_embedding

class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


def make_gin_conv(input_dim: int, out_dim: int) -> GINConv:
    mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim))
    return GINConv(mlp)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(make_gin_conv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        g1 = global_add_pool(h1, batch)
        h1_pred = self.predictor(h1_online)
        g2 = global_add_pool(h2, batch)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)
            g1_target = global_add_pool(h1_target, batch)
            g2_target = global_add_pool(h2_target, batch)

        return g1, g2, h1_pred, h2_pred, g1_target, g2_target


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    total_loss = 0
    x = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32).to(data.batch.device)

        optimizer.zero_grad()
        _, _, h1_pred, h2_pred, g1_target, g2_target = encoder_model(data.x, data.edge_index, batch=data.batch)

        loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred,
                              g1_target=g1_target.detach(), g2_target=g2_target.detach(), batch=data.batch)
        loss.backward()
        optimizer.step()
        encoder_model.update_target_encoder(0.99)
        x.append(g1_target)
        total_loss += loss.item()    
    x = torch.cat(x, dim=0)
    return total_loss,x


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        g1, g2, _, _, _, _ = encoder_model(data.x, data.edge_index, batch=data.batch)
        z = torch.cat([g1, g2], dim=1)
        x.append(z)
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
        aug1 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.1)])
        aug2 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.1)])
    
        gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32).to(device)
        contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='G2L').to(device)
    
        optimizer = Adam(encoder_model.parameters(), lr=0.01)
    
        with tqdm(total=1000, desc='(T)') as pbar:
            for epoch in range(1, 101):
                loss,g = train(encoder_model, contrast_model, dataloader, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()
    
        torch.save(g,'NASA_EMBEDDING_BGRL.pt')        
        #test_result = test(encoder_model, dataloader)
        #print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
        
        labels = np.loadtxt('labels_NASA.txt')
        
        labels =torch.tensor( labels)
        labels =np.array(labels.detach().cpu().numpy())
        
        emb = np.array(torch.load('NASA_EMBEDDING_BGRL.pt').detach().cpu().numpy())
        ans[i] = evaluate_embedding(emb,labels)[0]
    np.mean(ans)
    np.std(ans)

if __name__ == '__main__':
    main()
