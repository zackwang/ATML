import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh


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
(row-)normalization of input feature vectors described in G&B(2010)
"""
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

"""
Renormalization of adjacent matrix  
input: A
output: D^{-1/2}(A)D^{-1/2}
"""
def normalize_adj(mx):
    D = np.array(mx.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = sp.diags(D_inv_sqrt)
    mx = mx.dot(D_mat_inv_sqrt).transpose().dot(D_mat_inv_sqrt).tocoo()
    return mx

"""
Normalization of adjacent matrix based on chabyshev polynomials
input: adjacent matrix (adj), chebyshev order (k)
output: [t_0, ..., t_k] (t_i is sparse matric)
"""
def chebyshev_polynomials(adj, k):
    laplacian = sp.eye(adj.shape[0]) - normalize_adj(adj)
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t = [sp.eye(adj.shape[0]), scaled_laplacian]

    for i in range(2, k+1):
        s_lap = sp.csr_matrix(scaled_laplacian, copy=True)
        t_k = 2 * s_lap.dot(t[-1]) - t[-2]
        t.append(t_k)

    return t

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

"""
Convert a scipy sparse matrix to a torch sparse tensor.
"""
def convert_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
