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

def row_normalize(mtx):
    # normalize feature matrix by each row
    r_sum = np.array(mtx.sum(axis=1))
    r_sum_mtx = r_sum * np.ones(mtx.shape)
    normalized_feature_mtx = np.divide(mtx.todense(), r_sum_mtx)

    return normalized_feature_mtx

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
    features = row_normalize(features_mtx = sp.csr_matrix(content[:, 1:-1], dtype=np.float32))
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
    features = row_normalize(features_mtx = sp.csr_matrix(content[:, 1:-1], dtype=np.float32))
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
        lines = f.readlines() # skip first line

        # skip first line
        lines.pop(0)

        # build feature dict
        line = lines.pop(0)
        all_words = line.split()
        all_words = all_words[1:-1]

        feature_idx_map = {} # feature to idx mapping
        for i, token in enumerate(all_words):
            word = token.split(':')[1]
            feature_idx_map[word] = i
        
        row = 0
        nodes_idx_map = {}
        label_strings = []
        feature_matrix = sp.lil_matrix((len(lines), len(feature_idx_map)), dtype=np.float32)

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


def load_pubmed():
    path = './data/pubmed'

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


def load_dataset(dataset):
    if dataset == 'cora':
        return load_cora()
    elif dataset == 'citeseer':
        return load_citeseer()
    elif dataset == 'pubmed':
        return load_pubmed()