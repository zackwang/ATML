import numpy as np
import scipy.sparse as sp
import torch

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

def build_feature_matrix(content):
    features_mtx = sp.csr_matrix(content[:, 1:-1], dtype=np.float32)

    r_sum = np.array(features_mtx.sum(axis=1))
    r_sum_mtx = r_sum * np.ones(features_mtx.shape)

    return np.divide(features_mtx.todense(), r_sum_mtx)

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

    adj = sp.coo_matrix((matrix_data, (to_nodes, from_nodes)), shape=(num_nodes, num_nodes))

    # make adjacency matrix symmetric
    sym_adj = adj + adj.T.multiply(adj.T > adj)

    return sym_adj.tocoo()


def load_cora():
    path = './data/cora'

    content = np.genfromtxt(f"{path}/cora.content", dtype=np.dtype(str))
    cites = np.genfromtxt(f"{path}/cora.cites", dtype=np.int32)

    nodes = np.array(content[:, 0], dtype=np.int32)
    features = build_feature_matrix(content)
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

def load_citeseer():
    path = './data/citeseer'

    content = np.genfromtxt(f"{path}/citeseer.content", dtype=np.dtype(str))
    cites = np.genfromtxt(f"{path}/citeseer.cites", dtype=np.dtype(str))

    nodes = np.array(content[:, 0], dtype=np.dtype(str))
    features = build_feature_matrix(content)
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

def load_pubmed():
    pass

def load_dataset(dataset):
    if dataset == 'cora':
        return load_cora()
    elif dataset == 'citeseer':
        return load_citeseer()
    elif dataset == 'pubmed':
        return load_pubmed()