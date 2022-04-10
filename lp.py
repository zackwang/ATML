import argparse
import numpy as np
from torch_geometric.nn.models.label_prop import LabelPropagation
from torch_geometric.datasets import NELL, Planetoid, Coauthor

NUM_TRAIN_DATA_PER_LABEL_CLASS = 20
NUM_VAL_DATA = 500
NUM_TEST_DATA = 1000

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LP')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--num-layers', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    if args.dataset == 'cora':
        dataset = Planetoid(root='./data/cora', name='Cora')
    elif args.dataset == 'citeseer':
        dataset = Planetoid(root='./data/citeseer', name='CiteSeer')
    elif args.dataset == 'pubmed':
        dataset = Planetoid(root='./data/pubmed', name='PubMed')
    elif args.dataset == 'nell':
        dataset = NELL(root='./data/nell')
    elif args.dataset == 'coauthor':
        dataset = Coauthor(root='./data/coauthor', name='CS')
    else:
        raise Exception(f'unrecognized dataset: {args.dataset}')
    
    model = LabelPropagation(num_layers=args.num_layers, alpha=args.alpha)
    data = dataset.data

    if args.dataset == 'coauthor':
        # special case for coauthor dataset which does not have masks
        idx_train = np.array([], dtype=np.int32)
        labels_copy = np.copy(data.y.numpy())
        label_set = set(labels_copy)
        # sample 20 training dataset for each label class
        for label in label_set:
            sample = (np.where(labels_copy == label)[0]
                      [0:NUM_TRAIN_DATA_PER_LABEL_CLASS])
            idx_train = np.concatenate([idx_train, sample])
            labels_copy[sample] = -1
        # construct validation dataset
        idx_val = np.where(labels_copy != -1)[0][0: NUM_VAL_DATA]
        labels_copy[idx_val] = -1
        # construct testing dataset
        idx_test = np.where(labels_copy != -1)[0][0: NUM_TEST_DATA]
        train_mask = sample_mask(idx_train, labels_copy.shape[0])
        test_mask = sample_mask(idx_test, labels_copy.shape[0])
    else:
        train_mask = data.train_mask
        test_mask = data.test_mask

    
    pred = model(data.y, data.edge_index, train_mask).argmax(dim=1)
    correct = (pred[test_mask] == data.y[test_mask]).sum()
    acc = int(correct) / int(test_mask.sum())
    print(f'Accuracy: {acc:.4f}')