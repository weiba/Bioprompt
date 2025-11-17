from datetime import datetime
from sklearn import metrics
import torch
import torch.nn.functional as F
import argparse
import yaml
from yaml import SafeLoader
from time import perf_counter as t
import numpy as np
from mngcl import MNGCL
from sklearn import linear_model
import warnings
from gcn import GCN
from torch_geometric.utils import dropout_adj
from utils import fixSeed, dataLoader

warnings.filterwarnings("ignore")
cross_val = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(mask):
    model.train()
    optimizer.zero_grad()
    x = data.x
    ppiAdj_index = ppiAdj.coalesce().indices()
    pathAdj_index = pathAdj.coalesce().indices()
    goAdj_index = goAdj.coalesce().indices()
    # feature mask
    x_1 = F.dropout(x, drop_feature_rate_1)
    x_2 = F.dropout(x, drop_feature_rate_2)
    x_3 = F.dropout(x, drop_feature_rate_3)
    # edge dropout
    ppiAdj_index = dropout_adj(ppiAdj_index, p=drop_edge_rate_1, force_undirected=True)[0]
    pathAdj_index = dropout_adj(pathAdj_index, p=drop_edge_rate_2, force_undirected=True)[0]
    goAdj_index = dropout_adj(goAdj_index, p=drop_edge_rate_3, force_undirected=True)[0]
    pred1, pred2, pred3, _, conloss = model(ppiAdj_index, pathAdj_index, goAdj_index, x_1, x_2, x_3)
    loss1 = F.binary_cross_entropy_with_logits(pred1[mask], Y[mask])
    loss2 = F.binary_cross_entropy_with_logits(pred2[mask], Y[mask])
    loss3 = F.binary_cross_entropy_with_logits(pred3[mask], Y[mask])
    crloss = LAMBDA * (loss1 + loss2 + loss3)
    loss = (1 - 3 * LAMBDA) * conloss + crloss
    loss.backward()
    optimizer.step()
    return loss.item()

def LogReg(train_x, train_y, test_x):
    regr = linear_model.LogisticRegression(max_iter=10000)
    regr.fit(train_x, train_y.ravel())
    pre = regr.predict_proba(test_x)
    pre = pre[:, 1]
    return pre

@torch.no_grad()
def test(mask1, mask2):
    model.eval()
    ppiAdj_index = ppiAdj.coalesce().indices()
    pathAdj_index = pathAdj.coalesce().indices()
    goAdj_index = goAdj.coalesce().indices()
    _, _, _, emb, _ = model(ppiAdj_index, pathAdj_index, goAdj_index, data.x, data.x, data.x)
    train_x = torch.sigmoid(emb[mask1]).cpu().detach().numpy()
    train_y = Y[mask1].cpu().numpy()
    test_x = torch.sigmoid(emb[mask2]).cpu().detach().numpy()
    Yn = Y[mask2].cpu().numpy()
    pred = LogReg(train_x, train_y, test_x)
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(Yn, pred), area

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CPDB')
    parser.add_argument('--cancer_type', type=str, default='pan-cancer')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    dataPath = "data/" + args.dataset + "/"
    cancerType = args.cancer_type
    seed = config['seed']
    LR = config['LR']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_edge_rate_3 = config['drop_edge_rate_3']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    drop_feature_rate_3 = config['drop_feature_rate_3']
    tau = config['tau']
    EPOCH = config['EPOCH']
    LAMBDA = config['LAMBDA']

    fixSeed(seed)

    data, y_train, Y, y_all, mask_all, ppiAdj, pathAdj, goAdj, posList \
        = dataLoader(args, dataPath, cancerType, device)

    AUC = np.zeros(shape=(cross_val, 5))
    AUPR = np.zeros(shape=(cross_val, 5))
    train_time = t()

    if cancerType == 'pan-cancer':
        # pan-cancer
        print("---------Pan-cancer Train begin--------")
        gcn = GCN(data.x.shape[1], 300, 100).to(device)
        model = MNGCL(gnn=gcn,
                      posList=posList,
                      tau=tau,
                      gnn_outsize=100,
                      projection_hidden_size=300,
                      projection_size=100
                      ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        for train_epoch in range(1, EPOCH):
            # test
            loss = train(data.mask)
            test_AUC, test_AUPR = test(data.mask, data.mask_te)
            print(
                f'(T) | Epoch={train_epoch:03d}, test_loss={loss:.4f},test_AUC={test_AUC:.4f}, test_AUPR={test_AUPR:.4f}')
            now = t()
    else:
        # specific cancer
        print("---------" + cancerType + " cancer Train begin--------")
        path = dataPath + "Specific cancer/"
        mask_data = torch.load(path + cancerType + '.pkl')
        Y = mask_data[0]
        train_mask = mask_data[1]
        test_mask = mask_data[2]
        gcn = GCN(data.x.shape[1], 150, 50).to(device)
        model = MNGCL(gnn=gcn,
                      posList=posList,
                      tau=tau,
                      gnn_outsize=50,
                      projection_hidden_size=150,
                      projection_size=50
                      ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        for train_epoch in range(1, EPOCH):
            # test
            loss = train(train_mask)
            test_AUC, test_AUPR = test(train_mask, test_mask)
            print(
                f'(T) | Epoch={train_epoch:03d}, test_loss={loss:.4f}, test_AUC={test_AUC:.4f}, test_AUPR={test_AUPR:.4f}')
            now = t()

    with open('result/' + cancerType + '_test_result.txt', 'a') as f:
        print("-------------------------result------------------------", file=f)
        print("AUC:{:.4f}".format(test_AUC), file=f)
        print("AUPR:{:.4f}".format(test_AUPR), file=f)
        print("LR:{}".format(LR), file=f)
        print("Tau:{}".format(tau), file=f)
        print("EPOCH:{}".format(EPOCH), file=f)
        print("LAMBDA:{}".format(LAMBDA), file=f)
        print("PPI_mask:{}     x_mask:{}".format(drop_edge_rate_1, drop_feature_rate_1), file=f)
        print("Pathway_mask:{} x_mask:{}".format(drop_edge_rate_2, drop_feature_rate_2), file=f)
        print("GO_mask:{}      x_mask:{}".format(drop_edge_rate_3, drop_feature_rate_3), file=f)
        now_time = datetime.now()
        print("train_date:{}".format(now_time), file=f)
        print("spend time: {:.1f} S".format(now - train_time), file=f)
        print("Save to txt successly!")



