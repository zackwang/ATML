from pathlib import Path
from re import I

import random
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Coauthor, NELL, Planetoid
from utils import convert_tensor

NUM_TRAIN_DATA_PER_LABEL_CLASS = 20
NUM_VAL_DATA = 500
NUM_TEST_DATA = 1000


def encode_labels(label_strings):
    label_idx_map = {}
    counter = 0

    encoded_labels = []
    for label in label_strings:
        if label not in label_idx_map:
            label_idx_map[label] = counter
            counter += 1
        encoded_labels.append(label_idx_map[label])

    return np.array(encoded_labels, dtype=np.int32)


def row_normalize(mtx):
    # calculate a row sum array
    rsum = np.array(mtx.sum(1), dtype=np.float32).flatten()
    # calcuate it's inverse
    rsum_inv = np.power(rsum, -1, out=rsum, where=rsum != 0.)
    # create a diagnal matrix with inverse row sum value
    rsum_inv_diag_mtx = sp.diags(rsum_inv)
    # multiply each element by the corresponding row sum
    features = rsum_inv_diag_mtx.dot(mtx)

    return features


def build_adj_matrix(edges, nodes_idx_map):
    to_nodes = []
    from_nodes = []

    for e in edges:
        to_node, from_node = e
        if to_node in nodes_idx_map and from_node in nodes_idx_map:
            to_nodes.append(nodes_idx_map[to_node])
            from_nodes.append(nodes_idx_map[from_node])

    num_edges = len(to_nodes)
    num_nodes = len(nodes_idx_map)

    matrix_data = np.ones(num_edges, dtype=np.float32)
    to_nodes = np.array(to_nodes, dtype=np.int32)
    from_nodes = np.array(from_nodes, dtype=np.int32)

    adj = sp.coo_matrix((matrix_data, (to_nodes, from_nodes)),
                        shape=(num_nodes, num_nodes))

    # make adjacency matrix symmetric
    sym_adj = adj + adj.T.multiply(adj.T > adj)

    return sym_adj.tocoo()


def load_cora_from_raw():
    path = './data/cora_raw'

    content = np.genfromtxt(f"{path}/cora.content", dtype=np.dtype(str))
    cites = np.genfromtxt(f"{path}/cora.cites", dtype=np.int32)

    nodes = np.array(content[:, 0], dtype=np.int32)
    features = row_normalize(sp.csr_matrix(content[:, 1:-1], dtype=np.float32))
    labels = encode_labels(content[:, -1])

    # build graph
    nodes_idx_map = {node: i for i, node in enumerate(nodes)}

    adj_mtx = build_adj_matrix(cites, nodes_idx_map)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj_mtx, features, labels, idx_train, idx_val, idx_test


def load_citeseer_from_raw():
    path = './data/citeseer_raw'

    content = np.genfromtxt(f"{path}/citeseer.content", dtype=np.dtype(str))
    cites = np.genfromtxt(f"{path}/citeseer.cites", dtype=np.dtype(str))

    nodes = np.array(content[:, 0], dtype=np.dtype(str))
    features = row_normalize(sp.csr_matrix(content[:, 1:-1], dtype=np.float32))
    labels = encode_labels(content[:, -1])

    # build graph
    nodes_idx_map = {node: i for i, node in enumerate(nodes)}

    adj_mtx = build_adj_matrix(cites, nodes_idx_map)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj_mtx, features, labels, idx_train, idx_val, idx_test


def build_pubmed_features(path):
    with open(f"{path}/Pubmed-Diabetes.NODE.paper.tab") as f:
        lines = f.readlines()  # skip first line

        # skip first line
        lines.pop(0)

        # build feature dict
        line = lines.pop(0)
        all_words = line.split()
        all_words = all_words[1:-1]

        feature_idx_map = {}  # feature to idx mapping
        for i, token in enumerate(all_words):
            word = token.split(':')[1]
            feature_idx_map[word] = i

        row = 0
        nodes_idx_map = {}
        label_strings = []
        feature_matrix = sp.lil_matrix(
            (len(lines), len(feature_idx_map)), dtype=np.float32)

        for line in lines:
            # format: 3542527	label=2	w-use=0.027970030654407077	w-studi=0.013917168664368762	summary=w-use,w-studi
            if not line.strip():
                continue
            tokens = line.split()

            node = tokens[0]
            assert(node not in nodes_idx_map)
            nodes_idx_map[node] = row

            label = tokens[1].split('=')[1]
            label_strings.append(label)

            features = tokens[2:-1]
            for feature in features:
                fname, fvalue = feature.split('=')
                feature_matrix[(row, feature_idx_map[fname])] = fvalue

            row += 1

        features = row_normalize(feature_matrix)
        labels = encode_labels(label_strings)

        return features, labels, nodes_idx_map


def build_pubmed_adj_matrix(path, nodes_idx_map):
    with open(f"{path}/Pubmed-Diabetes.DIRECTED.cites.tab") as f:
        lines = f.readlines()

        # skip first two lines
        lines.pop(0)
        lines.pop(0)

        edges = []
        for line in lines:
            # format: 33824	paper:19127292	|	paper:17363749
            if not line.strip():
                continue

            tokens = line.split()
            from_node = tokens[1].split(':')[1]
            to_node = tokens[-1].split(':')[1]
            edges.append([to_node, from_node])

        return build_adj_matrix(edges, nodes_idx_map)


def load_pubmed_from_raw():
    path = './data/pubmed_raw'

    features, labels, nodes_idx_map = build_pubmed_features(path)
    adj_mtx = build_pubmed_adj_matrix(path, nodes_idx_map)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj_mtx, features, labels, idx_train, idx_val, idx_test


def edge_index_to_adj_mtx(edge_index, num_nodes):
    num_edges = edge_index.shape[1]

    matrix_data = np.ones(num_edges, dtype=np.float32)
    from_nodes = edge_index[0]
    to_nodes = edge_index[1]

    adj = sp.coo_matrix((matrix_data, (to_nodes, from_nodes)),
                        shape=(num_nodes, num_nodes))

    # make adjacency matrix symmetric
    sym_adj = adj + adj.T.multiply(adj.T > adj)

    return sym_adj


def load_from_torch_geometric(dataset_name, mode):
    Path(f"./data/{dataset_name}").mkdir(parents=True, exist_ok=True)
    if dataset_name == 'cora':
        dataset = Planetoid(root='./data/cora', name='Cora')
    elif dataset_name == 'citeseer':
        dataset = Planetoid(root='./data/citeseer', name='CiteSeer')
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root='./data/pubmed', name='PubMed')
    elif dataset_name == 'nell':
        dataset = NELL(root='./data/nell')
    elif dataset_name == 'coauthor':
        dataset = Coauthor(root='./data/coauthor', name='CS')
    else:
        raise Exception(f'unrecognized dataset: {dataset}')

    # get lable vector
    labels = dataset.data.y
    num_label_classes = len(set(labels.numpy()))

    # build feature matrix
    features_tensor = dataset.data.x
    if hasattr(features_tensor, 'to_scipy'):
        # special case for pubmed dataset
        features_mtx = features_tensor.to_scipy()
    else:
        features_mtx = sp.csr_matrix(features_tensor.numpy(), dtype=np.float32)

    features_mtx = row_normalize(features_mtx)

    values = features_mtx.data
    features_mtx = features_mtx.tocoo()
    indices = np.vstack((features_mtx.row, features_mtx.col))

    features = torch.sparse.FloatTensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        features_mtx.shape)

    # build adjacency matrix
    num_nodes = labels.shape[0]
    edge_index_tensor = dataset.data.edge_index
    edge_index = edge_index_tensor.numpy()
    adj_mtx = edge_index_to_adj_mtx(edge_index, num_nodes)

    # build training, validation and test set index
    NUM_TRAIN_DATA_PER_LABEL_CLASS = 20
    # use official data split
    if mode == 'official':
        if dataset_name == 'coauthor':
            # special case for coauthor dataset which does not have masks
            idx_train = np.array([], dtype=np.int32)
            labels_copy = np.copy(labels.numpy())
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
        else:
            idx_train = np.where(dataset.data.train_mask.numpy())[0]
            idx_val = np.where(dataset.data.val_mask.numpy())[0]
            idx_test = np.where(dataset.data.test_mask.numpy())[0]

        if dataset_name != 'nell':
            assert(len(idx_train) == NUM_TRAIN_DATA_PER_LABEL_CLASS * num_label_classes)
            assert(len(idx_val) == NUM_VAL_DATA)
            assert(len(idx_test) == NUM_TEST_DATA)
        
        idx_train = [idx_train]
        idx_val = [idx_val]
        idx_test = [idx_test]
    # use random split for ten times
    elif mode == 'random':
        if dataset_name == 'nell':
            NUM_TRAIN_DATA_PER_LABEL_CLASS = 2
        idx_train = list()
        idx_val = list()
        idx_test = list()
        for i in range(10):
            labels_copy = np.copy(labels.numpy())
            label_set = set(labels_copy)
            idx_train_0 = np.array([], dtype=np.int32)

            # sample 20 training dataset for each label class
            for label in label_set:
                whole = np.where(labels_copy == label)[0]
                start = random.randint(0, whole.shape[0]-NUM_TRAIN_DATA_PER_LABEL_CLASS)
                sample = whole[start:start+NUM_TRAIN_DATA_PER_LABEL_CLASS]
                idx_train_0 = np.concatenate([idx_train_0, sample])
                labels_copy[sample] = -1
            
            idx_val_0 = np.where(labels_copy != -1)[0][0: NUM_VAL_DATA]
            labels_copy[idx_val_0] = -1
            # construct testing dataset
            idx_test_0 = np.where(labels_copy != -1)[0][0: NUM_TEST_DATA]

            idx_train.append(idx_train_0)
            idx_val.append(idx_val_0)
            idx_test.append(idx_test_0)

    elif mode == 'fiveFold':
        idx_train = list()
        idx_val = list()
        idx_test = list()
        fold_size = int(num_nodes/5)
        for i in range(5):
            if i < 2:
                idx_train_0 = [*range(i*fold_size, (i+4)*fold_size)]
                idx_test_0 = [*range((i+4)*fold_size, num_nodes)] + [*range(0, i*fold_size)]
            else:
                idx_train_0 = [*range(i*fold_size, num_nodes)] + [*range(0, (i-1)*fold_size)]
                idx_test_0 = [*range((i-1)*fold_size, i*fold_size)]
            idx_val_0 = idx_train_0[0:fold_size]
            
            idx_train.append(idx_train_0)
            idx_val.append(idx_val_0)
            idx_test.append(idx_test_0)
            # import pdb; pdb.set_trace()

    idx_train = [torch.LongTensor(_) for _ in idx_train]
    idx_val = [torch.LongTensor(_) for _ in idx_val]
    idx_test = [torch.LongTensor(_) for _ in idx_test]

    return adj_mtx, features, labels, idx_train, idx_val, idx_test


def load_dataset(dataset, mode):
    return load_from_torch_geometric(dataset, mode)

def generate(n):
    adj = sp.csr_matrix((n,n), dtype=np.float32).tolil()
    edges = set()
    while len(edges) < 2*n:
        a = random.randint(0,n-1)
        b = random.randint(0,n-1)
        if a == b:
            continue
        if (a,b) in edges:
            continue
        adj[a, b] = 1
        adj[b, a] = 1
        edges.add((a,b))
        edges.add((b,a))
    adj_mtx = adj.tocoo()

    features = sp.eye(n)
    features = convert_tensor(features)

    labels = torch.LongTensor(np.ones(n))
    idx_train = [torch.LongTensor([*range(n)])]
    idx_val = [torch.LongTensor([])]
    idx_test = [torch.LongTensor([1])]
    
    print("Generation finished!")
    return adj_mtx, features, labels, idx_train, idx_val, idx_test



        
