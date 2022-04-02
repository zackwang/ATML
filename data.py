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


def build_adj_matrix(edges, num_nodes):
    num_edges = edges.shape[0]
    matrix_data = np.ones(num_edges, dtype=np.float32)

    to_nodes = edges[:, 0]
    from_nodes = edges[:, 1]

    adj = sp.coo_matrix((matrix_data, (to_nodes, from_nodes)), shape=(num_nodes, num_nodes))

    # make adjacency matrix symmetric
    sym_adj = adj + adj.T.multiply(adj.T > adj)

    # conver to pytorch sparse tensor
    sparse_mtx = sym_adj.tocoo()
    indices = torch.from_numpy(np.array([sparse_mtx.row, sparse_mtx.col]).astype(np.int64))
    values = torch.from_numpy(sparse_mtx.data)
    shape = torch.Size(sparse_mtx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_cora():
    path = './data/cora'

    content = np.genfromtxt(f"{path}/cora.content", dtype=np.dtype(str))
    cites = np.genfromtxt(f"{path}/cora.cites", dtype=np.int32)

    nodes = np.array(content[:, 0], dtype=np.int32)
    features = np.array(content[:, 1:-1], dtype=np.float32)
    labels = encode_labels(content[:, -1])

    # build graph
    num_nodes = nodes.shape[0]
    nodes_idx_map = {j: i for i, j in enumerate(nodes)}
    edges = np.array([nodes_idx_map[node] for node in cites.flatten()], dtype=np.int32).reshape(cites.shape)

    adj_mtx = build_adj_matrix(edges, num_nodes)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj_mtx, features, labels, idx_train, idx_val, idx_test
