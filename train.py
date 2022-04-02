from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, preprocess_adj
from models import GCN, MultiValGCN

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
adj, features, labels, idx_train, idx_val, idx_test = load_data()
preprocessed = preprocess_adj(adj, args.propogation, args.order)

# Model and optimizer
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

optimizer = optim.Adam([
                        {'params': model.gc1.parameters()},
                        {'params': model.gc2.parameters(), 'weight_decay':0}
                    ], lr=args.lr, weight_decay=args.weight_decay)


def train():
    model.train()
    best_loss_val = 0
    early_stop_loss_val = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = model(features, preprocessed)
        loss = F.nll_loss(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        if best_loss_val == 0 or best_loss_val > loss_val:
            best_loss_val = loss_val

        if (epoch+1)%10 == 0:
            if args.early_stopping and early_stop_loss_val == best_loss_val:
                break
            early_stop_loss_val = best_loss_val

            acc_train = accuracy(output[idx_train], labels[idx_train])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val),
                'acc_val: {:.4f}'.format(acc_val.item()))


def test():
    model.eval()
    output = model(features, preprocessed)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
train()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
