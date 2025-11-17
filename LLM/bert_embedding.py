import numpy as np
import pandas as pd
import time
import pickle
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
import ollama
import numpy as np
import pandas as pd
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pytorch_transformers import BertModel
import nltk
from nltk.tokenize import WordPunctTokenizer
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=16, sparsity_param=0.05, beta=3):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_param = sparsity_param
        self.beta = beta

    def forward(self, x):
        hidden = torch.sigmoid(self.encoder(x))
        output = torch.sigmoid(self.decoder(hidden))
        
        return output, hidden

    def sparsity_loss(self, hidden):
        mean_activation = torch.mean(hidden, dim=0)
        kl_divergence = self.sparsity_param * torch.log(self.sparsity_param / mean_activation) + \
                        (1 - self.sparsity_param) * torch.log((1 - self.sparsity_param) / (1 - mean_activation))
        return self.beta * torch.sum(kl_divergence)

def sparse_autoencoder_loss(output, target, hidden, sparsity_loss_func):
    # BCE = F.binary_cross_entropy(output, target, reduction='sum')
    BCE = F.mse_loss(output, target, reduction='sum')   
    sparsity = sparsity_loss_func(hidden)
    return BCE + sparsity

class FNClassifier(nn.Module):
    def __init__(self, model_name="LLM/biobert-base-cased-v1.2"):
        super(FNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.autoencoder = SparseAutoencoder()

    def forward(self, x, return_latent=False):
        reconstructed, hidden = self.autoencoder(x)
        if return_latent:
            return hidden
        return reconstructed, hidden

def train(model, data_loader, optimizer):
    model.train()
    train_loss = 0
    for _, data in enumerate(data_loader):
        # print(data)
        optimizer.zero_grad()
        output, hidden = model(data)
        loss = sparse_autoencoder_loss(output, data, hidden, model.autoencoder.sparsity_loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(data_loader.dataset)


def load_label_single(path, cancerType):
    label = np.loadtxt(path + "label_file-P-"+cancerType+".txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.loadtxt(path + "pos-"+cancerType+".txt", dtype=int)
    label_neg = np.loadtxt(path + "neg.txt", dtype=int)
    return Y, label_pos, label_neg

def split_newmask(path, cancerType):
    label, label_pos, label_neg = load_label_single(path)
    tr_mask , te_mask = train_test_split(np.arange(len(label)), test_size=0.2, random_state=42)
    return tr_mask, te_mask

device = torch.device('cuda')
def get_bert_output(mask_all, dataset, model, feature, device):
    y = torch.zeros(len(mask_all), 768).to(device)
    num_features = 6
    weights = torch.linspace(1.0, 0.05, steps=num_features).to(device)
    print('weights:', weights)
    for i in range(len(mask_all)):
        # if mask_all[i]:
        if i % 100.0 == 0:
            print(i,'/',len(mask_all))
        # print(feature.iloc[i]['GeneName'])
        tokenized_features = []
        for j in range(1, 7):
            if feature.iloc[i][f'feature{j}'] != 'none':
                tokens = model.tokenizer.tokenize(str(feature.iloc[i][f'feature{j}']))
                tokenized_features.append(model.tokenizer.convert_tokens_to_ids(tokens))
            else:
                tokenized_features.append([])


        tokens_tensors = [torch.tensor(tokens, dtype=torch.long) if tokens else torch.zeros(1, dtype=torch.long) for tokens in tokenized_features]
        tokens_tensor = pad_sequence(tokens_tensors, batch_first=True).to(device)
        

        pooled_output = model.bert(tokens_tensor)[1]
        
        adjusted_weights = torch.tensor([weights[j] if feature.iloc[i][f'feature{j+1}'] != 'none' else 0 for j in range(num_features)], device=device)
        
        weighted_pooled_output = torch.sum(pooled_output * adjusted_weights.unsqueeze(1), dim=0) / adjusted_weights.sum()
        y[i] = weighted_pooled_output
    torch.save(y, f'/home/yuantao/code/LLM/bert_out/{dataset}-weight.pt')
    return y

# datasets = ['kirc', 'brca', 'prad', 'stad', 'hnsc', 'luad', 'thca', 'blca', 'esca', 'lihc', 'ucec']
llm_name = 'gemma2'
datasets = ['pan-cancer']
for dataset in datasets:
    model = FNClassifier()
    model.to(device)
    output_dir = r'/home/yuantao/code/LLM/csv/{}'.format(llm_name)
    output_file = r'/home/yuantao/code/LLM/csv/gemma2/PAN-CANCER_go_descibe.csv'
    if not os.path.exists(output_dir):
        print(f"no such directory: {output_dir}")
        continue

    if os.path.exists(f'/home/yuantao/code/LLM/bert_out/{dataset}.pt'):
        # y = torch.load(f'/home/yuantao/code/LLM/bert_out/{dataset}.pt')
        dataset = dataset.upper()
        csvpath = '/home/yuantao/code/LLM/csv/' + dataset + '-6.csv'
        feature = pd.read_csv(csvpath, header=None, names=['ENSG', 'GeneName', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6'],encoding='utf-8')
        mask_all = len(feature) * [True]
        print('mask_all:', mask_all)
        y = get_bert_output(mask_all, dataset, model, feature, device)
    else:
        dataset = dataset.upper()
        csvpath = '/home/yuantao/code/LLM/csv/' + dataset + '-6.csv'
        feature = pd.read_csv(csvpath, header=None, names=['ENSG_GeneName', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6'],encoding='utf-8')
        mask_all = len(feature) * [True]
        print('mask_all:', mask_all)
        y = get_bert_output(mask_all, dataset, model, feature, device)