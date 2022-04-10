# Reproduction on Graph Convolutional Networks

## Requirements

- Python >= 3.7

- PyTorch, torch_geometric, numpy, scipy, sklearn

## Datasets

We use dataset `Cora`, `Citeseer`, `Pubmed`, `Coauthor` and `NELL` preprocessed by `torch_geometric`.

## GCN

To train GCN model, use `python train.py --model GCN --dataset dataset_name` to run experiments on given datasets.

- --dataset, dataset name to run, can be `cora`, `citeseer`, `pubmed`, `coauthor` and `nell`.

- --propagation, propagation method to run, default is `Renomalization`. Choices can be `Chebyshev`, (should specify order with --order), `FirstOrder`,  `SingleParam`, `Renormalization`, `FirstOrderOnly`, `MLP`

- --mode, mode of dataset split, default is `official`. Choices can be `official` (official split), `random` (will run experiments with 10 random splits), or `fiveFold` (five-fold cross validation).

To enable early stopping, please add `--early_stopping`.

You can also specify hyper parameters with `--runs`, `--epochs`, `--lr`, `--dropout`, `--weight_decay`, `--hidden`. Default setting is for training on citation networks. To train model on NELL, please change it to `--weight_decay 1e-5 --dropout 0.1 --hidden 64`.

## GCN-LPA

To train GCN-LPA model, use `python train.py --model GCN-LPA  --dataset dataset_name ` to run experiments on given datasets.

- --dataset, dataset name to run, can be `cora`, `citeseer`, `pubmed`, `coauthor` and `nell`.

- --layers, number of GCN-LPA layers, default is 2

- --LPA_iteration, number of iterations in LPA, default is 5.

- --labmda, weight between GCN loss and LPA loss, default is 5.

To enable early stopping, please add `--early_stopping`.

You can also specify hyper parameters with `--runs`, `--epochs`, `--lr`, `--dropout`, `--weight_decay`, `--hidden`.

## GAT

To train GAT model, use `python train.py --model GAT --dataset dataset_name` to run experiments on given datasets.

- --dataset, dataset name to run, can be `cora`, `citeseer`, `pubmed`, `coauthor` and `nell`.

- --GAT_nb_heads, number of head attentions in GAT, default is 8

- --GAT_alpha, alpha in LeakyReLU, defalt is 0.2

To enable early stopping, please add `--early_stopping`.

You can also specify hyper parameters with `--runs`, `--epochs`, `--lr`, `--dropout`, `--weight_decay`, `--hidden`.

## Baseline

### Label Propagation

To train model on LP, please run `python lp.py --dataset dataset_name`. You can also specify hyper parameters by `--num-layers` or `--alpha`.

### Label Propagation Algorithm

To  train model on LPA, please run `python lpa.py --dataset dataset_name`. You can also specify hyper parameters by `--iter` or `--ratio`.

### DeepWalk

To  train model on LPA, please run `python deepwalk.py --dataset dataset_name`.
