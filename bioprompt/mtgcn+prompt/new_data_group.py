import os
import torch
import random
import warnings
import numpy as np
from random import shuffle
import torch.optim as optim
from sklearn.manifold import TSNE
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
from collections import deque
import pickle as pk
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.nn import Linear
from torch_scatter import scatter_mean
from torch_geometric.utils import degree
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.nn import ChebConv, SAGEConv, GINConv, GatedGraphConv
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops,add_self_loops
from collections import deque




@torch.no_grad()
def test(mask):
    model.eval()
    x, _, _, _ = model()

    pred = torch.sigmoid(x[mask]).cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()
    predicted_classes = (pred >= 0.5).astype(int)  
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    accuracy = np.mean(predicted_classes == Yn)
    return metrics.roc_auc_score(Yn, pred), area


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ChebConv(64, 300, K=2, normalization="sym")
        self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
        self.conv3 = ChebConv(100, 1, K=2, normalization="sym")

        self.lin1 = Linear(64, 100)
        self.lin2 = Linear(64, 100)

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self):
        edge_index, _ = dropout_adj(data.edge_index, p=0.5,
                                    force_undirected=True,
                                    num_nodes=data.x.size()[0],
                                    training=self.training)

        x0 = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()

        neg_edge_index = negative_sampling(pb, 13627, 504378)

        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()

        r_loss = pos_loss + neg_loss


        x = F.dropout(x1, training=self.training)
        x = self.conv3(x, edge_index)

        return x,  r_loss,self.c1, self.c2



EPOCH = 2500
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))
# 忽略警告
warnings.filterwarnings("ignore")

data = torch.load(r"/home/yuantao/code/MTGCN-master/data/CPDB_new_data.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
data = data.to(device)
Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
y_all = np.logical_or(data.y, data.y_te)
mask_all = np.logical_or(data.mask, data.mask_te)
data.x = data.x[:, :48]

datas = torch.load(r"/home/yuantao/code/MTGCN-master/str_fearures.pkl")
data.x = torch.cat((data.x, datas), 1)
data = data.to(device)

with open("/home/yuantao/code/MTGCN-master/data/k_sets.pkl", 'rb') as handle:
    k_sets = pickle.load(handle)

pb, _ = remove_self_loops(data.edge_index)
pb, _ = add_self_loops(pb)
E = data.edge_index


def found_max_group_feature_random_start(data, label, maxsize=5, tr_mask=None, device='cuda'):
    data = data.to(device)
    edge_index = data.edge_index.t().to(device)
    y = torch.tensor(data.y).to(device)
    # if tr_mask is not None:
    #     tr_mask = tr_mask.to(device)

    visited = set()
    max_group = []
    max_edges = []

    def bfs(start):
        queue = deque([start])
        component = []
        edges = []
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                component.append(node)
                if len(component) >= maxsize:
                    return component, edges
                # 在GPU上查找相邻节点
                for edge in edge_index.cpu().numpy():  # CPU上的numpy操作
                    if edge[0] == node:
                        neighbor = edge[1]
                    elif edge[1] == node:
                        neighbor = edge[0]
                    else:
                        continue
                    if neighbor not in visited and y[neighbor] == label and (tr_mask is None or tr_mask[neighbor]):
                        queue.append(neighbor)
                        edges.append(edge)
        return component, edges

    potential_starts = [i for i in range(len(y)) if y[i] == label and (tr_mask is None or tr_mask[i])]
    random.shuffle(potential_starts)

    for start in potential_starts:
        if start not in visited:
            group, edges = bfs(start)
            if len(group) > len(max_group):
                max_group, max_edges = group, edges
                if len(max_group) >= maxsize:
                    break

    max_group_feature = data.x[max_group].to(device)  # 确保节点特征数据在GPU上
    subgraph_edges = torch.tensor(max_edges, device=device)  # 将找到的边信息转移到GPU

    # 创建子图
    subgraph = Data(x=max_group_feature, edge_index=subgraph_edges)

    return subgraph


def loss_max_group_feature(maxsize=4, tr_mask=None, margin=0.3):
    group_pos = found_max_group_feature_random_start(data, 1, maxsize, tr_mask)
    group_neg = found_max_group_feature_random_start(data, 0, maxsize, tr_mask)
    group_pos = group_pos.to(device)
    group_neg = group_neg.to(device)

    print(group_pos.size, group_neg.size)
    # loss = F.binary_cross_entropy_with_logits(group_pos, group_neg)

    group_pos_feature = model(group_pos)
    group_neg_feature = model(group_neg)

    # 创建匹配的 target 张量
    target = torch.full((group_pos.size(0),), -1, dtype=torch.float, device=device)

    # 计算 Cosine Embedding Loss
    cosine_loss = nn.CosineEmbeddingLoss(margin=margin)
    loss = cosine_loss(group_pos, group_neg, target)

    return loss
    


def train(mask):
    model.train()
    optimizer.zero_grad()
    pred, rl, c1, c2 = model()


    loss1 = F.binary_cross_entropy_with_logits(pred[mask], Y[mask]) / (c1 * c1) + rl / (c2 * c2) + 2 * torch.log(c2 * c1)
    loss2 = loss_max_group_feature(4, mask)
    loss = loss1 + loss2
    # print("loss:", loss)
    # print("loss1:", loss1)
    # print("loss2:", loss2)
    # print()

    # loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask]) / (c1 * c1) + rl / (c2 * c2) + 2 * torch.log(c2 * c1)

    loss.backward()
    optimizer.step()
EPOCH = 2500
for i in range(10):
    for cv_run in range(5):
        print(i, cv_run)
        _, _, tr_mask, te_mask = k_sets[i][cv_run]
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, EPOCH):
            train(tr_mask)
            if epoch % 100 == 0:
                print(epoch)

        AUC[i][cv_run], AUPR[i][cv_run] = test(te_mask)
        print(AUC[i][cv_run], AUPR[i][cv_run])

    mean_auc = np.mean(AUC)
    var_auc = np.var(AUC)
    mean_aupr = np.mean(AUPR)
    var_aupr = np.var(AUPR)
    # mean_acc = np.mean(ACC)
    # var_acc = np.var(ACC)

    # Save results to file with precision of four decimal places
    np.savetxt(r'result\\auc_results1.txt', AUC, fmt='%.4f')
    np.savetxt(r'result\\aupr_results1.txt', AUPR, fmt='%.4f')
    # np.savetxt(r'result\\acc_results1.txt', ACC, fmt='%.4f')

    print("Mean AUC: {:.4f}".format(mean_auc))
    print("Variance AUC: {:.4f}".format(var_auc))
    print("Mean AUPR: {:.4f}".format(mean_aupr))
    print("Variance AUPR: {:.4f}".format(var_aupr))
    # print("Mean ACC: {:.4f}".format(mean_acc))
    # print("Variance ACC: {:.4f}".format(var_acc))