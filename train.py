import time
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from utils import preprocess_adj, one_hot_embedding, convert_tensor
from models import GCN, MultiValGCN, GCNLPA, GAT
from data import load_dataset, generate

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GCN",
                        choices=["GCN", "GCN-LPA", "GAT"],
                        required=False, help='Model to train.')
    parser.add_argument('--layers', type=int, default=5, 
                        required=False, help='Number of layers.')
    parser.add_argument('--dataset', type=str, default="cora",
                        choices=["cora", "citeseer", "pubmed", "nell", "coauthor"],
                        required=True, help='Dataset to conduct experiments.')
    parser.add_argument('--propogation', type=str, default="Renormalization",
                        choices=["Chebyshev", "FirstOrder", "SingleParam", "Renormalization", "FirstOrderOnly", "MLP"],
                        required=False, help='Propogation method.')
    parser.add_argument('--order', type=int, default=1,
                        required=False, help='Chebyshev order.')
    parser.add_argument('--LPA_iteration', type=int, default=5, 
                        required=False, help='LPA iteration')
    parser.add_argument('--LPA_lambda', type=float, default=5,
                        help='Weight between GCN loss and LPA loss.')
    parser.add_argument('--GAT_nb_heads', type=int, default=8, 
                        required=False, help='Number of head attentions in GAT.')
    parser.add_argument('--mode', type=str, default="official",
                        choices=["official", "fiveFold", "random", "generate"],
                        required=False, help='Mode of data split.')
    parser.add_argument('--runs', type=int, default=1, 
                        required=False, help='runs of random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        required=False, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        required=False, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        required=False, help='Weight decay for L2 regularization (only for the first layer).')
    parser.add_argument('--hidden', type=int, default=16,
                        required=False, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        required=False, help='Dropout rate.')
    parser.add_argument('--GAT_alpha', type=float, default=0.2, 
                        required=False, help='Alpha for the leaky_relu in GAT.')
    parser.add_argument('--early_stopping', action='store_true', default=False,
                        help='Enables early stopping with window size of 10.')

    args = parser.parse_args()

    # Load data
    if args.mode != "generate":
        adj, features, labels, idx_train, idx_val, idx_test = load_dataset(args.dataset, args.mode)
    else:
        adj, features, labels, idx_train, idx_val, idx_test = generate(5000)
    
    # preprocessing
    if args.model == "GCN":
        preprocessed = preprocess_adj(adj, args.propogation, args.order)
    elif args.model == "GCN-LPA":
        adj = convert_tensor(adj).to_dense()
        features = features.to_dense()
        labels_for_lpa = one_hot_embedding(labels)
    elif args.model == "GAT":
        preprocessed = preprocess_adj(adj, "Renormalization", 0).to_dense()
        features = features.to_dense()
    else:
        print("No model is found!")
        exit()

    runs = list()
    for seed in range(args.runs):
        np.random.seed(seed)
        torch.manual_seed(seed)

        results = list()
        # train model with different splits
        for i in range(len(idx_train)):

            # Model
            if args.model == "GCN":
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
            elif args.model == "GCN-LPA":
                model = GCNLPA(input=features.shape[1],
                            hidden=args.hidden,
                            output=labels.max().item() + 1,
                            adj=adj,
                            dropout=args.dropout,
                            layer_num=args.layers,
                            lpa_iter = args.LPA_iteration)
            elif args.model == "GAT":
                model = GAT(input=features.shape[1], 
                            hidden=args.hidden, 
                            output=labels.max().item() + 1, 
                            dropout=args.dropout, 
                            nheads=args.GAT_nb_heads, 
                            alpha=args.GAT_alpha)

            # optimizer and criterion
            if args.model == "GCN":
                optimizer = optim.Adam([
                                        {'params': model.gc1.parameters(), },
                                        {'params': model.gc2.parameters(), 'weight_decay': 0},
                                    ], lr=args.lr, weight_decay=args.weight_decay)
                #criterion = nn.NLLLoss()
            else:
                optimizer = optim.Adam(model.parameters(),
                                        lr=args.lr, weight_decay=args.weight_decay)
            
            criterion = nn.NLLLoss()

            # Train
            t = time.time()
            model.train()
            best_loss_val = 0
            early_stop_loss_val = 0
            for epoch in range(args.epochs):
                optimizer.zero_grad()
                
                if args.model in ["GCN", "GAT"]:
                    output = model(features, preprocessed)
                    loss = criterion(output[idx_train[i]], labels[idx_train[i]])
                elif args.model == "GCN-LPA":
                    output, y_hat = model(features, labels_for_lpa)
                    loss = criterion(output[idx_train[i]], labels[idx_train[i]]) + args.LPA_lambda * criterion(y_hat[idx_train[i]], labels[idx_train[i]])
                loss.backward()
                optimizer.step()

                loss_val = criterion(output[idx_val[i]], labels[idx_val[i]]).item()

                if best_loss_val == 0 or best_loss_val > loss_val:
                    best_loss_val = loss_val

                if (epoch+1)%10 == 0:
                    if args.early_stopping and early_stop_loss_val == best_loss_val:
                        break
                    early_stop_loss_val = best_loss_val

            print("Total time: {:.4f}s".format(time.time() - t))

            # Test
            model.eval()
            if args.model in ["GCN", "GAT"]:
                output = model(features, preprocessed)
            elif args.model == "GCN-LPA":
                output, _ = model(features, labels_for_lpa)
            preds = output.max(1)[1].type_as(labels)
            acc_test = accuracy_score(labels[idx_test[i]], preds[idx_test[i]])
            print("Test set results:", "accuracy= {:.4f}".format(acc_test))
            results.append(acc_test)

        if len(results) > 1:
            results = np.array(results)
            average = results.mean()
            print("Average accuracy among {} splits: {} +/- {}".format(len(idx_train), average, results.std()))
            runs.append(average)
        else:
            runs.append(results[0])

    if len(runs) > 1:
        runs = np.array(runs)
        print("Average accuracy among {} runs: {} +/- {}".format(args.runs, runs.mean(), runs.std()))