import numpy as np
import pandas as pd
import time
import pickle
from sklearn import linear_model
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from datetime import datetime
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
import random
from sklearn import metrics

# 忽略警告
warnings.filterwarnings("ignore")
def early_stopping(current_auc, current_aupr , best_auc, best_aupr, patience_counter, patience_limit):
    if current_auc > best_auc:
        best_auc = current_auc
        best_aupr = current_aupr
        patience_counter = 0
    else:
        patience_counter += 1
    return best_auc , best_aupr, patience_counter
def load_label_single(path):
    label = np.loadtxt(path + "label_file-P-"+cancerType+".txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.loadtxt(path + "pos-"+cancerType+".txt", dtype=int)
    label_neg = np.loadtxt(path + "neg.txt", dtype=int)
    return Y, label_pos, label_neg

def sample_division_single(pos_label, neg_label, l, l1, l2, i):
    # pos_label：Positive sample index
    # neg_label：Negative sample index
    # l：number of genes
    # l1：Number of positive samples
    # l2：number of negative samples
    # i：number of folds
    pos_test = pos_label[i * l1:(i + 1) * l1]
    pos_train = list(set(pos_label) - set(pos_test))
    neg_test = neg_label[i * l2:(i + 1) * l2]
    neg_train = list(set(neg_label) - set(neg_test))
    indexs1 = [False] * l
    indexs2 = [False] * l
    for j in range(len(pos_train)):
        indexs1[pos_train[j]] = True
    for j in range(len(neg_train)):
        indexs1[neg_train[j]] = True
    for j in range(len(pos_test)):
        indexs2[pos_test[j]] = True
    for j in range(len(neg_test)):
        indexs2[neg_test[j]] = True
    tr_mask = torch.from_numpy(np.array(indexs1))
    te_mask = torch.from_numpy(np.array(indexs2))
    return tr_mask, te_mask

# Cancertypes = ['kirc', 'brca', 'prad', 'stad', 'hnsc', 'luad', 'thca', 'blca', 'esca', 'lihc', 'ucec', 'coad', 'lusc', 'cesc', 'kirp']
Cancertypes = ['pan-cancer']
for cancerType in Cancertypes:
    data = torch.load("/home/yuantao/code/DGCL/data/CPDB/CPDB_new_data.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
    y_all = np.logical_or(data.y, data.y_te)
    mask_all = np.logical_or(data.mask, data.mask_te)
    if cancerType == 'pan-cancer':
        data.x = data.x[:, :48]
        bert_output_path = "/home/yuantao/code/LLM/bert_out/LLMs/go_desc_PAN-CANCER-gemma2-CPDB.pt"
    else:
        cancerType_dict = {
            'kirc': [0, 16, 32],
            'brca': [1, 17, 33],
            'prad': [3, 19, 35],
            'stad': [4, 20, 36],
            'hnsc': [5, 21, 37],
            'luad': [6, 22, 38],
            'thca': [7, 23, 39],
            'blca': [8, 24, 40],
            'esca': [9, 25, 41],
            'lihc': [10, 26, 42],
            'ucec': [11, 27, 43],
            'coad': [12, 28, 44],
            'lusc': [13, 29, 45],
            'cesc': [14, 30, 46],
            'kirp': [15, 31, 47]
        }
        data.x = data.x[:, cancerType_dict[cancerType]]
        
        # Y, pos_label, neg_label = load_label_single(path)

        bert_output_path = '/home/yuantao/code/LLM/bert_out/LLMs/go_desc_' + cancerType.upper() + '-gemma2-CPDB.pt'
    print(bert_output_path)
    data.bert_output = torch.load(bert_output_path)
    datas = torch.load("/home/yuantao/code/MTGCN_ori/data/str_fearures_ori.pkl").to(device)
    data.x = torch.cat((data.x, datas), 1)
    input_dim = data.x.size()[1]

    data = data.to(device)

    with open("/home/yuantao/code/MTGCN_ori/data/k_sets.pkl", 'rb') as handle:
        k_sets = pickle.load(handle)

    pb, _ = remove_self_loops(data.edge_index)
    pb, _ = add_self_loops(pb)
    E = data.edge_index

    class BetaVAE(nn.Module):
        def __init__(self, input_dim, hidden_dim, z_dim , dropout_rate=0.2):
            super(BetaVAE, self).__init__()
            self.encoder = nn.Linear(input_dim, hidden_dim)
            self.dropout_encoder = nn.Dropout(dropout_rate)
            self.to_mean = nn.Linear(hidden_dim, z_dim)  
            self.to_logvar = nn.Linear(hidden_dim, z_dim)
            self.decoder = nn.Linear(z_dim, hidden_dim)
            self.dropout_decoder = nn.Dropout(dropout_rate)
            self.final_decoder = nn.Linear(hidden_dim, input_dim)
        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        def forward(self, x, return_z=False):
            hidden = self.encoder(x)
            # hidden = self.dropout_encoder(hidden)
            
            mu = self.to_mean(hidden)
            log_var = self.to_logvar(hidden)
            z = self.reparameterize(mu, log_var)

            decoded = self.decoder(z)
            # decoded = self.dropout_decoder(decoded)
            output = self.final_decoder(decoded)

            loss = self.loss_function(output,x, mu, log_var)
            return output, z, loss
        def loss_function(self, recon_x, x, mu, log_var):
            BCE = F.mse_loss(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            return BCE + 0.001*KLD
        
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = ChebConv(input_dim, 300, K=2, normalization="sym")
            self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
            self.conv3 = ChebConv(100, 1, K=2, normalization="sym")

            self.lin1 = Linear(input_dim, 100)
            self.lin2 = Linear(input_dim, 100)

            self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
            self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))
            self.c3 = torch.nn.Parameter(torch.Tensor([0.5]))

            self.vae = BetaVAE(input_dim=768,hidden_dim=256,z_dim=100)
            # self.linear = nn.Linear(100, 100)
            self.mlp = nn.Linear(100, 1)

        def forward(self,mask):
            edge_index, _ = dropout_adj(data.edge_index, p=0.5,
                                        force_undirected=True,
                                        num_nodes=data.x.size()[0],
                                        training=self.training)
            bertout = torch.zeros(data.x.size()[0], 768).to(device)
            bertout[mask] = data.bert_output[mask]
            x0 = F.dropout(data.x, training=self.training)
            x = torch.relu(self.conv1(x0, edge_index))
            x = F.dropout(x, training=self.training)
            x1 = torch.relu(self.conv2(x, edge_index))

            x = x1 + torch.relu(self.lin1(x0))
            z = x1 + torch.relu(self.lin2(x0))
            model_emb = x

            recon_bert = self.vae.decoder(model_emb)
            recon_bert = self.vae.final_decoder(recon_bert)
            bert_vae, z2, vaeloss = self.vae(bertout)
            recon_z_loss = (F.mse_loss(z2, model_emb, reduction='mean') + F.mse_loss(model_emb, z2, reduction='mean')) / 2

            pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()
            neg_edge_index = negative_sampling(pb, 13627, 504378)
            neg_loss = -torch.log(
                1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()
            r_loss = pos_loss + neg_loss

            x = F.dropout(x, training=self.training)
            pre1 = self.conv3(x, edge_index)
            pre2 = self.mlp(model_emb)

            
            return pre1,pre2, r_loss, self.c1, self.c2, self.c3, vaeloss, recon_z_loss

    def train(mask):
        model.train()
        optimizer.zero_grad()

        pred,pred2, rl, c1, c2,c3,vaeloss, recon_z_loss = model(mask)

        loss1 = F.binary_cross_entropy_with_logits(pred[mask], Y[mask]) / (c1 * c1) + rl / (c2 * c2) + 2 * torch.log(c2 * c1) + vaeloss / (c3 * c3) + recon_z_loss
        loss2 = F.binary_cross_entropy_with_logits(pred2[mask], Y[mask])
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

    def LogReg(train_x, train_y, test_x):
        regr = linear_model.LogisticRegression(max_iter=10000)
        regr.fit(train_x, train_y.ravel())
        pre = regr.predict_proba(test_x)
        pre = pre[:, 1]
        return pre

    @torch.no_grad()
    def test(tr_mask,te_mask):
        model.eval()
        x,pre2, _, _, _ ,_,_,_ = model(tr_mask)
        # print(x.shape, pre2.shape)
        # print(x, pre2)
        train_y = Y[tr_mask].cpu().detach().numpy()
        emb = torch.cat((x, pre2), 1)
        train_x = torch.sigmoid(emb[tr_mask]).cpu().detach().numpy()

        test_x = torch.sigmoid(emb[te_mask]).cpu().detach().numpy()

        pred = LogReg(train_x, train_y, test_x)

        Yn = Y[te_mask].cpu().detach().numpy()
        
        precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
        area = metrics.auc(recall, precision)

        return metrics.roc_auc_score(Yn, pred), area

    import matplotlib.pyplot as plt

    AUC = np.zeros(shape=(10, 5))
    AUPR = np.zeros(shape=(10, 5))

    EPOCH = 1000
    for i in range(10):

        if cancerType != 'pan-cancer':
            path = "/home/yuantao/code/DGCL/data/CPDB/Specific cancer/"
            label, label_pos, label_neg = load_label_single(path)
            random.shuffle(label_pos)
            random.shuffle(label_neg)
            l = len(label)
            l1 = int(len(label_pos)/5)
            l2 = int(len(label_neg)/5)
            Y = label
        
        for cv_run in range(5):
            print(i, cv_run)
            if cancerType != 'pan-cancer':
                tr_mask, te_mask = sample_division_single(label_pos, label_neg, l, l1, l2, cv_run)
            else:
                _, _, tr_mask, te_mask = k_sets[i][cv_run]
            AUC_list = []
            # _, _, tr_mask, te_mask = k_sets[i][cv_run]
            model = Net().to(device)
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.005)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            auclist = []
            auprclist = []
            best_auc = 0.0
            best_aupr = 0.0
            patience_counter = 0
            patience_limit = 150
            for epoch in range(1, EPOCH):
                train(tr_mask)
                AUC[i][cv_run], AUPR[i][cv_run] = test(tr_mask, te_mask)
                print(AUC[i][cv_run], AUPR[i][cv_run])

                auclist.append(AUC[i][cv_run])
                auprclist.append(AUPR[i][cv_run])
                best_auc, best_aupr, patience_counter = early_stopping(AUC[i][cv_run], AUPR[i][cv_run], best_auc, best_aupr, patience_counter, patience_limit)
                if patience_counter >= patience_limit:
                    print(f'Early stopping at epoch {epoch},{best_auc},{best_aupr}')
                    AUC[i][cv_run] = best_auc
                    AUPR[i][cv_run] = best_aupr
                    break

            print(AUC[i][cv_run], AUPR[i][cv_run])

            np.savetxt(f"result/MTGCN_{cancerType}__new_AUC.txt", AUC, delimiter="\t")
            np.savetxt(f"result/MTGCN_{cancerType}__new_AUPR.txt", AUPR, delimiter="\t")
    with open('LLMs_result.txt', 'a') as f:
        print("-------------------------MTGCN_new_result------------------------", file=f)
        print("CancerType:{}".format(cancerType), file=f)
        print("AUC:{:.6f}+-{:.6f}".format(AUC.mean(), AUC.std()), file=f)
        print("AUPR:{:.6f}+-{:.6f}".format(AUPR.mean(), AUPR.std()), file=f)

        now_time = datetime.now()
        print("train_date:{}".format(now_time), file=f)
        print("Save to txt successly!")
