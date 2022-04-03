import time
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from utils import preprocess_adj
from models import GCN, MultiValGCN
from data import load_citeseer, load_cora

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CORA",
                    choices=["CORA", "Citeseer", "Pubmed"],
                    required=True, help='Dataset to conduct experiments.')
parser.add_argument('--propogation', type=str, default="Renormalization",
                    choices=["Chebyshev", "FirstOrder", "SingleParam", "Renormalization", "FirstOrderOnly", "MLP"],
                    required=True, help='Propogation method.')
parser.add_argument('--order', type=int, default=1,
                    required=False, help='Chebyshev order.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay for L2 regularization (only for the first layer).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate.')
parser.add_argument('--early_stopping', action='store_true', default=False,
                    help='Enables early stopping with window size of 10.')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_cora()
preprocessed = preprocess_adj(adj, args.propogation, args.order)

# Model
if args.propogation in ["SingleParam", "Renormalization", "FirstOrderOnly", "MLP"]:
    model = GCN(input=features.shape[1],
            hidden=args.hidden,
            output=labels.max().item() + 1,
            dropout=args.dropout)

elif args.propogation in ["Chebyshev", "FirstOrder"]:
    model = MultiValGCN(input=features.shape[1],
            hidden=args.hidden,
            output=labels.max().item() + 1,
            order=args.order+1,
            dropout=args.dropout)

# optimizer and criterion
optimizer = optim.Adam([
                        {'params': model.gc1.parameters()},
                        {'params': model.gc2.parameters(), 'weight_decay':0}
                    ], lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.NLLLoss()

def train():
    model.train()
    best_loss_val = 0
    early_stop_loss_val = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = model(features, preprocessed)
        loss = criterion(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        loss_val = criterion(output[idx_val], labels[idx_val]).item()
        if best_loss_val == 0 or best_loss_val > loss_val:
            best_loss_val = loss_val

        if (epoch+1)%10 == 0:
            if args.early_stopping and early_stop_loss_val == best_loss_val:
                break
            early_stop_loss_val = best_loss_val

            preds = output.max(1)[1].type_as(labels)
            acc_train = accuracy_score(labels[idx_train], preds[idx_train])
            acc_val = accuracy_score(labels[idx_val], preds[idx_val])
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val),
                'acc_val: {:.4f}'.format(acc_val.item()))


def test():
    model.eval()
    output = model(features, preprocessed)
    loss_test = criterion(output[idx_test], labels[idx_test])
    preds = output.max(1)[1].type_as(labels)
    acc_test = accuracy_score(labels[idx_test], preds[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train
t = time.time()
train()
print("Total time: {:.4f}s".format(time.time() - t))

# Test
test()
