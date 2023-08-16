# Optional: eliminating warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import matplotlib.pyplot as plt
from arguments import arg_parse
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from evaluate_embedding import evaluate_embedding
from gin import Encoder
from losses import local_global_loss_
from model import FF, PriorDiscriminator
from torch import optim
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import json
import json
import numpy as np
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import os

class InfoGraph(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(InfoGraph, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)

    if self.prior: # default is False
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def forward(self, x, edge_index, batch, num_graphs):
    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    x = x.to(torch.float32) # have to convert torch.float64 to torch.float32
    y, M = self.encoder(x, edge_index, batch)

    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR

if __name__ == '__main__':
    args = arg_parse()
    accuracies = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}
    epochs = 1000
    log_interval = 1
    batch_size = 32
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = []
    files = [filename for filename in sorted(os.listdir('FIRE_D_Superpixel_Graphs'), key=len) if filename.startswith("superpixel_graph_")]  # 386; 21460
    files = sorted(files)
    for i in range(len(files)):
        print("i is:", i)
        tmp = torch.load('FIRE_D_Superpixel_Graphs/' + files[i])
        dataset.append(tmp)
    dataset_num_features = dataset[0].x.size(1)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InfoGraph(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader)

    # k-means
    num_clusters = 2

    print('===== Before training =====')

    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)))
        # log_interval = 1
        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader)

            np.savez_compressed('graphcl_emb' + str(epoch), emb)


    with open('unsupervised.log', 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s))
