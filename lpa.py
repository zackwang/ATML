import argparse
import numpy as np
import torch
from random import sample
import random
from sklearn.metrics import accuracy_score

from data import load_dataset
from utils import convert_tensor

def preprocess_adj(adj):
    mask = torch.where(adj>0, adj, -9e9*torch.ones_like(adj))
    adj = torch.softmax(mask, dim=1)
    return adj

def train(adj, train_idx, labels, iter):
    nb_classes = max(labels) + 1
    y = list()
    mask = list()
    true_row = [True for _ in range(nb_classes)]
    false_row = [False for _ in range(nb_classes)]

    for i in range(len(labels)):
        letter = [0 for _ in range(nb_classes)]
        if(i in train_idx):
            letter[labels[i]] = 1
            mask.append(true_row)
        else:
            mask.append(false_row)
        y.append(letter)    
            

    mask = torch.BoolTensor(mask)
    y_o = torch.FloatTensor(y)
    y = torch.FloatTensor(y)
    for _ in range(int(iter)):
        # Y^{l+1} = D^{-1}A Y^l
        y = torch.mm(adj, y)
        # y_i^{l+1} = y_i^{l}
        y = torch.where(mask, y_o, y)

    return y

random.seed(234)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora", help='Dataset name (cora, citeseer, pubmed).')
parser.add_argument('--iter', type=float, default=20, help='LPA iteration')
parser.add_argument('--ratio', type=float, default=-1, help='ratio between test and train')

args = parser.parse_args()

# Load Data
adj, features, labels, idx_train, idx_val, idx_test = load_dataset(args.dataset, 'official')

# Convert Adjacency Matrix To the Desired Form
adj = preprocess_adj(convert_tensor(adj).to_dense())

# Create Training and Testing Index
if(args.ratio != -1):   # customered index
    train_idx = sample(list(np.arange(len(labels))), int(len(labels) * args.ratio))
    test_idx = [item for item in [*range(0, len(labels))] if item not in train_idx]
else:                   # default idx
    train_idx = idx_train[0]
    test_idx = idx_test[0]

# Start Training
y = train(adj, train_idx, labels, args.iter)
preds = y.max(1)[1].type_as(labels)
acc_test = accuracy_score(labels[idx_test[0]], preds[idx_test[0]])
print("Test set results:", acc_test)

        