import pickle
from datetime import datetime
from sklearn import metrics
import random
import torch
import torch.backends.cudnn as cudnn

import argparse
import yaml
from yaml import SafeLoader
import numpy as np

def fixSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CPDB')
    parser.add_argument('--cancer_type', type=str, default='pan-cancer')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--config', type=str, default='config1.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    dataPath = "data/" + args.dataset + "/"
    cancerType = args.cancer_type
    seed = config['seed']
    return args, config, dataPath, cancerType, seed

def dataLoader(args,dataPath,cancerType,device):
    data = torch.load(dataPath + args.dataset + "_data.pkl")
    data = data.to(device)
    y_train = data.y
    Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
    y_all = np.logical_or(data.y, data.y_te)
    mask_all = np.logical_or(data.mask, data.mask_te)
    if cancerType == 'pan-cancer':
        data.x = data.x[:, :48]
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
    print(data.x)
    # node2VEC feature
    dataz = torch.load(dataPath + "Str_feature.pkl")
    dataz = dataz.to(device)
    # 64D feature
    data.x = torch.cat((data.x, dataz), 1)
    #network
    ppiAdj = torch.load(dataPath + 'ppi.pkl')
    ppiAdj_self = torch.load(dataPath + 'ppi_selfloop.pkl')
    pathAdj = torch.load(dataPath + 'pathway_SimMatrix.pkl')
    goAdj = torch.load(dataPath + 'GO_SimMatrix.pkl')
    #pos
    pos1 = ppiAdj_self.to_dense()
    pos2 = pathAdj.to_dense() + torch.eye(data.x.shape[0]).cuda()
    pos3 = goAdj.to_dense() + torch.eye(data.x.shape[0]).cuda()
    posList = [pos1, pos2, pos3]
    return data, y_train, Y, y_all, mask_all, ppiAdj, pathAdj, goAdj, posList
