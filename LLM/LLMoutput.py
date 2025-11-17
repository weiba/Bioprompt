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
import os
from datetime import datetime
def is_valid_output(feature):
    return len(feature.split()) < 12 and feature != ''

EPOCH = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prompt = open("LLM/txt/prompt/prompt_new_go.txt", "r",encoding='gbk').read()
cancer_name_txt = pd.read_csv("LLM/txt/data/cancer.txt", header=None).values
sentance = pd.read_csv(r'LLM/csv/finally_df.csv')
sentance.columns = ['Gene_Symbol','Ensembl_ID','Description']

cancer_names = [ 'pan-cancer']
modelname = 'llama3.1'


for cancer_name in cancer_names:
    cancer_name = cancer_name.upper()
    print(f"Processing {cancer_name}...")
    cancer_name_prompt = cancer_name
    for row in cancer_name_txt:
        if row[0].startswith(cancer_name_prompt):
            cancer_name_prompt = row[0]
            cancer_name_prompt = cancer_name_prompt.replace("\t"
            , "_")
            break
    new_prompt = prompt.replace('Cancer_name', cancer_name_prompt)
    # print(new_prompt)
    gene_name_list = open(r'/home/yuantao/code/LLM/txt/data/node_names.txt','r',encoding='utf-8').read().split('\n')

    output_dir = '/home/yuantao/code/LLM/csv/' + modelname + '/'
    output_file = os.path.join(output_dir, f"{cancer_name}_go_descibe.csv")
    MAX_FEATURES = 6
    
    f1 = open(output_file, "a")
    
    if os.path.exists(output_file):
        print(f"{output_file} exists, skipping...")
        with open(output_file, 'r') as f:
            lines = f.readlines()
            if len(lines)<len(gene_name_list):
                lenth = len(lines)
                # gene_name_list = gene_name_list[len(lines):]
            else:
                print('All done')
                continue
        print('begin from:', gene_name_list[lenth])
    
    for idx, gene_name in enumerate(gene_name_list):
        if idx < lenth:
            continue
        while True:
            new_prompt1 = new_prompt.replace('Gene_name', gene_name).replace('go_description', str(sentance['Description'][idx][:2048]))
            # print(new_prompt1)
            content = ollama.generate(model=modelname, prompt=new_prompt1)['response']

            start_pos = content.find('<Summary_b>')
            end_pos = content.rfind('<Summary_e>')
            
            if start_pos == -1 or end_pos == -1 or start_pos >= end_pos:
                # print(content)
                print("Invalid output format, retrying...")
                continue

            feature_start_idx = start_pos + len('<Summary_e>')
            feature_end_idx = end_pos
            feature = content[feature_start_idx:feature_end_idx].strip()
            

            feature_list = feature.split(',')
            if len(feature_list) > MAX_FEATURES: 
                feature_list = feature_list[:MAX_FEATURES]
            elif len(feature_list) < MAX_FEATURES:
                feature_list += ['None'] * (MAX_FEATURES - len(feature_list))
            
            # if is_valid_output(feature_list[0]):
            if all([is_valid_output(feature) for feature in feature_list]):
                feature_list = [word.strip().lower() for word in feature_list if word != '']
                print(f"gene_name: {gene_name}, Feature: {feature_list}")
                # print('processing...', idx+1)              
                break
            else:
                # print(feature)
                print("Invalid feature")
                print("Retrying...")


        f1.write(f"{gene_name},{','.join(feature_list)}\n")
        if idx%20 == 0:
            f1.flush()
            print(f"{idx+1}/{len(gene_name_list)}")
    f1.close()
