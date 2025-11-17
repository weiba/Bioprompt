import pickle
from datetime import datetime
from sklearn import metrics
import random
import torch
import torch.nn.functional as F
from time import perf_counter as t
import numpy as np
from mngcl import MNGCL
from sklearn import linear_model
import warnings
import gcnPreprocessing
from gcn import GCN
from torch_geometric.utils import dropout_adj
from utils import Config, fixSeed, dataLoader

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cross_val = 10

def load_label_single(path):
    label = np.loadtxt(path + "label_file-P-" + cancerType + ".txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.loadtxt(path + "pos-" + cancerType + ".txt", dtype=int)
    label_neg = np.loadtxt(path + "neg.txt", dtype=int)
    return Y, label_pos, label_neg

def sample_division_single(pos_label, neg_label, l, l1, l2, i):
    pos_val = pos_label[i * l1:(i + 1) * l1]
    pos_train = list(set(pos_label) - set(pos_val))
    neg_val = neg_label[i * l2:(i + 1) * l2]
    neg_train = list(set(neg_label) - set(neg_val))
    indexs1 = [False] * l
    indexs2 = [False] * l
    for j in range(len(pos_train)):
        indexs1[pos_train[j]] = True
    for j in range(len(neg_train)):
        indexs1[neg_train[j]] = True
    for j in range(len(pos_val)):
        indexs2[pos_val[j]] = True
    for j in range(len(neg_val)):
        indexs2[neg_val[j]] = True
    tr_mask = torch.from_numpy(np.array(indexs1))
    val_mask = torch.from_numpy(np.array(indexs2))
    return tr_mask, val_mask

def train(mask, LAMBDA):
    model.train()
    optimizer.zero_grad()
    x = data.x

    bert_output = torch.zeros(data.x.shape[0],768).to(device)
    bert_output[mask] = data.bert_output[mask]

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
    pred1, pred2, pred3, _, conloss,recon_z_loss,vaeloss,emb4 = model(ppiAdj_index, pathAdj_index, goAdj_index, x_1, x_2, x_3,bert_output)
    loss1 = F.binary_cross_entropy_with_logits(pred1[mask], Y[mask])
    loss2 = F.binary_cross_entropy_with_logits(pred2[mask], Y[mask])
    loss3 = F.binary_cross_entropy_with_logits(pred3[mask], Y[mask])
    loss4 = F.binary_cross_entropy_with_logits(emb4[mask], Y[mask])
    crloss = LAMBDA * (loss1 + loss2 + loss3 + loss4)
    loss = (1 - 4*LAMBDA) * conloss + crloss + recon_z_loss + 0.015 * vaeloss
    loss.backward()
    optimizer.step()
    return loss.item()

def LogReg(train_x, train_y, test_x):
    regr = linear_model.LogisticRegression(max_iter=10000)
    regr.fit(train_x, train_y.ravel())
    pre = regr.predict_proba(test_x)
    pre = pre[:, 1]
    return pre
def early_stopping(current_auc, current_aupr , best_auc, best_aupr, patience_counter, patience_limit):
    if current_auc > best_auc:
        best_auc = current_auc
        best_aupr = current_aupr
        patience_counter = 0
    else:
        patience_counter += 1
    return best_auc , best_aupr, patience_counter
@torch.no_grad()
def test(mask1, mask2):
    model.eval()
    ppiAdj_index = ppiAdj.coalesce().indices()
    pathAdj_index = pathAdj.coalesce().indices()
    goAdj_index = goAdj.coalesce().indices()
    _, _, _, emb, _ ,_,_,_ = model(ppiAdj_index, pathAdj_index, goAdj_index, data.x, data.x, data.x, data.bert_output)
    train_x = torch.sigmoid(emb[mask1]).cpu().detach().numpy()
    train_y = Y[mask1].cpu().numpy()
    test_x = torch.sigmoid(emb[mask2]).cpu().detach().numpy()
    Yn = Y[mask2].cpu().numpy()
    pred = LogReg(train_x, train_y, test_x)
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(Yn, pred), area


if __name__ == '__main__':
    bertout_types = ['go_llm_descibe']
    Cancertypes = ['pan-cancer']

    for cancerType in Cancertypes:
        for bertout_type in bertout_types:
            args, config, dataPath, cancerType, seed = Config(cancerType=cancerType)
            # Hyper-parameter setting
            LR = config['LR']
            drop_edge_rate_1 = config['drop_edge_rate_1']
            drop_edge_rate_2 = config['drop_edge_rate_2']
            drop_edge_rate_3 = config['drop_edge_rate_3']
            drop_feature_rate_1 = config['drop_feature_rate_1']
            drop_feature_rate_2 = config['drop_feature_rate_2']
            drop_feature_rate_3 = config['drop_feature_rate_3']
            tau = config['tau']
            tau = 0.7
            # EPOCH = 3
            patience_limit = 400
            EPOCH = config['EPOCH']
            LAMBDA = config['LAMBDA']

            fixSeed(seed)
            data, y_train, Y, y_all, mask_all, ppiAdj, pathAdj, goAdj, posList \
                = dataLoader(args, dataPath, cancerType, device)
            AUC = np.zeros(shape=(cross_val, 5))
            AUPR = np.zeros(shape=(cross_val, 5))
            train_time = t()
            if bertout_type == 'go_only':
                data.bert_output = torch.load('/home/yuantao/code/LLM/bert_out/GO_emb.pt')
            elif bertout_type == 'go_llm':
                data.bert_output = torch.load(f'/home/yuantao/code/LLM/bert_out/GO_LLM_{cancerType}_emb.pt')
            elif bertout_type == 'go_llm_descibe':
                data.bert_output = torch.load(f'/home/yuantao/code/LLM/bert_out/GO_LLM_{cancerType}_emb-descibe.pt')
            else:
                print("Wrong bertout_type!")
                exit()
            if cancerType == 'pan-cancer':
                # pan-cancer
                print("---------Pan-cancer Train begin--------")
                k_sets = gcnPreprocessing.cross_validation_sets(y=y_train, mask=data.mask)
                
                for i in range(cross_val):
                    for cv_run in range(5):
                        print("----------------------- i: %d, cv_run: %d -------------------------" % (i + 1, cv_run + 1))
                        start = t()
                        y_tr, y_val, tr_mask, val_mask = k_sets[i][cv_run]
                        gcn = GCN(data.x.shape[1], 300, 100).to(device)
                        model = MNGCL(gnn=gcn,
                                    posList=posList,
                                    tau=tau,
                                    gnn_outsize=100,
                                    projection_hidden_size=300,
                                    projection_size=100
                                    ).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                        auclist = []
                        auprclist = []
                        best_auc = 0.0
                        best_aupr = 0.0
                        patience_counter = 0
                        patience_limit = 150
                        for train_epoch in range(1, EPOCH):
                            loss = train(tr_mask, LAMBDA)
                            AUC[i][cv_run], AUPR[i][cv_run] = test(tr_mask, val_mask)
                            auclist.append(AUC[i][cv_run])
                            auprclist.append(AUPR[i][cv_run])

                            print(f'(T) | Epoch={train_epoch:03d}, val_loss={loss:.4f},val_AUC={AUC[i][cv_run]:.4f}, val_AUPR={AUPR[i][cv_run]:.4f}')
                            best_auc, best_aupr, patience_counter = early_stopping(AUC[i][cv_run], AUPR[i][cv_run], best_auc, best_aupr, patience_counter, patience_limit)
                            if patience_counter >= patience_limit:
                                print(f'Early stopping at epoch {train_epoch},{best_auc},{best_aupr}')
                                AUC[i][cv_run] = best_auc
                                AUPR[i][cv_run] = best_aupr
                                break
                        np.savetxt("result/MNGCL_AUC.txt", AUC, delimiter="\t")
                        np.savetxt("result/MNGCL_AUPR.txt", AUPR, delimiter="\t")
                        now = t()
                        print("this cv_run spend {:.1f}  s".format(now - start))
                    print('cv:', i + 1)
                    print("AUC:{:.4f}".format(AUC[i].mean()))
                    print("AUPR:{:.4f}".format(AUPR[i].mean()))
            else:
                # specific cancer
                print("---------" + cancerType + " Train begin--------")
                path = dataPath + "Specific cancer/"
                label, label_pos, label_neg = load_label_single(path)
                random.shuffle(label_pos)
                random.shuffle(label_neg)
                print(label)
                y_train_pos = label_pos[:int(0.75 * len(label_pos))]
                y_test_pos = label_pos[int(0.75 * len(label_pos)):]
                y_train_neg = label_neg[:int(0.75 * len(label_neg))]
                y_test_neg = label_neg[int(0.75 * len(label_neg)):]

                l = len(label)
                l1 = int(len(y_train_pos) / 5)
                l2 = int(len(y_train_neg) / 5)
                Y = label
                tr_mask, val_mask = sample_division_single(y_train_pos, y_train_neg, l, l1, l2, 0)
                train_mask = np.logical_or(tr_mask, val_mask)  # [0,0,0]
                train_mask = train_mask.bool()
                test_mask_index = np.concatenate((y_test_pos,y_test_neg))
                print('tets_mask:',len(test_mask_index))
                test_mask = [False] * l
                for j in range(len(test_mask_index)):
                    test_mask[test_mask_index[j]] = True
                Scancer_data = [Y,train_mask,test_mask]
                torch.save(Scancer_data,path+cancerType+'.pkl')
                
                for i in range(cross_val):
                    for cv_run in range(5):
                        print("----------------------- i: %d, cv_run: %d -------------------------" % (i + 1, cv_run + 1))
                        start = t()
                        tr_mask, val_mask = sample_division_single(y_train_pos, y_train_neg, l, l1, l2, cv_run)
                        print(tr_mask.shape)
                        print(tr_mask.sum())
                        print(val_mask.sum())
                        gcn = GCN(data.x.shape[1], 150, 50).to(device)
                        model = MNGCL(gnn=gcn,
                                    posList=posList,
                                    tau=tau,
                                    gnn_outsize=50,
                                    projection_hidden_size=150,
                                    projection_size=50
                                    ).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                        auclist = []
                        auprclist = []
                        best_auc = 0.0
                        best_aupr = 0.0
                        patience_counter = 0
                        patience_limit = 150
                        for train_epoch in range(1, EPOCH):
                            loss = train(tr_mask, LAMBDA)
                            AUC[i][cv_run], AUPR[i][cv_run] = test(tr_mask, val_mask)
                            auclist.append(AUC[i][cv_run])
                            auprclist.append(AUPR[i][cv_run])

                            print(f'(T) | Epoch={train_epoch:03d}, val_loss={loss:.4f},val_AUC={AUC[i][cv_run]:.4f}, val_AUPR={AUPR[i][cv_run]:.4f}')
                            best_auc, best_aupr, patience_counter = early_stopping(AUC[i][cv_run], AUPR[i][cv_run], best_auc, best_aupr, patience_counter, patience_limit)
                            if patience_counter >= patience_limit:
                                print(f'Early stopping at epoch {train_epoch},{best_auc},{best_aupr}')
                                AUC[i][cv_run] = best_auc
                                AUPR[i][cv_run] = best_aupr
                                break
                        np.savetxt("result/MNGCL_" + cancerType + "_AUC.txt", AUC, delimiter="\t")
                        np.savetxt("result/MNGCL_" + cancerType + "_AUPR.txt", AUPR, delimiter="\t")
                        now = t()
                        print("this cv_run spend {:.1f}  s".format(now - start))

            with open('result.txt', 'a') as f:
                print("-------------------------GO_CV_result------------------------", file=f)
                print("Feature:{}".format(bertout_type), file=f)
                print(cancerType,file=f)
                print("AUC:{:.6f}+-{:.6f}".format(AUC.mean(), AUC.std()), file=f)
                print("AUPR:{:.6f}+-{:.6f}".format(AUPR.mean(), AUPR.std()), file=f)

                # print("LR:{}".format(LR), file=f)
                # print("Tau:{}".format(tau), file=f)
                # print("EPOCH:{}".format(EPOCH), file=f)
                # print("LAMBDA:{}".format(LAMBDA), file=f)
                # print("PPI_mask:{}     x_mask:{}".format(drop_edge_rate_1, drop_feature_rate_1), file=f)
                # print("Pathway_mask:{} x_mask:{}".format(drop_edge_rate_2, drop_feature_rate_2), file=f)
                # print("GO_mask:{}      x_mask:{}".format(drop_edge_rate_3, drop_feature_rate_3), file=f)
                now_time = datetime.now()
                print("train_date:{}".format(now_time), file=f)
                # print("spend time: {:.1f} S".format(now - train_time), file=f)
                print("Save to txt successly!")