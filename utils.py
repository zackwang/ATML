import torch
import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features).tocoo().astype(np.float32)
    return convert_tensor(features)

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data0/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data0/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)
    labels = torch.LongTensor(labels)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


"""
Preprocess adjacent matrix for different propogation methods
input: adjacent matrix, propogation methods, order(used for Chebyshev method)
output: a list of sparse matrixes (for methods with multi variables) 
        a sparse matrix (for other methods)
"""
def preprocess_adj(adj, propogation, order):
    if propogation == "Renormalization":
        adj = adj + sp.eye(adj.shape[0])
        adj = normalize_adj(adj)
    elif propogation == "MLP":
        adj = None
    elif propogation in ["FirstOrder","FirstOrderOnly"]:
        adj = normalize_adj(adj)
    elif propogation == "SingleParam":
        adj = normalize_adj(adj)
        adj = adj + sp.eye(adj.shape[0])
    elif propogation == "Chebyshev":
        adj = chebyshev_polynomials(adj, order)
    
    if type(adj) is list:
        preprocessed = [convert_tensor(_) for _ in adj ]
    elif adj is not None:
        preprocessed = convert_tensor(adj)
    
    if propogation == "FirstOrder":
        preprocessed = [None, preprocessed]

    return preprocessed

"""
Renormalization of adjacent matrix  
input: A
output: D^{-1/2}(A)D^{-1/2}
"""
def normalize_adj(mx):
    D = np.array(mx.sum(axis=1)).flatten()
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt = np.diag(D_inv_sqrt)
    D_inv_sqrt = sp.coo_matrix(D_inv_sqrt)
    mx = mx.dot(D_inv_sqrt).transpose().dot(D_inv_sqrt)
    return mx

"""
Normalization of adjacent matrix based on chabyshev polynomials
input: adjacent matrix (adj), chebyshev order (k)
output: [t_0, ..., t_k] (t_i is sparse matric)
"""
def chebyshev_polynomials(adj, k):
    IN = sp.eye(adj.shape[0])
    laplacian = IN - normalize_adj(adj)
    eigenvalue, eigenvector = eigsh(laplacian, k=1, which='LM')
    rescaled_lap = (2.0 / eigenvalue[0]) * laplacian - IN

    t = [IN, rescaled_lap]

    for i in range(2, k+1):
        t_k = 2.0 * rescaled_lap.dot(t[-1]) - t[-2]
        t.append(t_k)

    return t


"""
Convert a sparse coo matrix to a torch sparse tensor.
"""
def convert_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    values = sparse_mx.data
    values = torch.FloatTensor(values)
    indices = np.vstack((sparse_mx.row, sparse_mx.col))
    indices = torch.LongTensor(indices)
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)