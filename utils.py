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
    else:
        preprocessed = adj
    
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
    D[np.where(D==0)] = 1
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt = sp.diags(D_inv_sqrt)
    # D_inv_sqrt = np.diag(D_inv_sqrt)
    # D_inv_sqrt = sp.coo_matrix(D_inv_sqrt)
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
    eigenvalue, eigenvector = eigsh(laplacian, k=1, which='LM', tol=1E-5)
    rescaled_lap = (2.0 / eigenvalue[0]) * laplacian - IN

    t = [IN, rescaled_lap]

    for i in range(2, k+1):
        t_k = 2.0 * rescaled_lap.dot(t[-1]) - t[-2]
        t.append(t_k)

    return t


"""
Convert a sparse coo matrix to a torch sparse tensor.
Input: a sparse matrix
Output: a torch sparse tensor
"""
def convert_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    values = sparse_mx.data
    values = torch.FloatTensor(values)
    indices = np.vstack((sparse_mx.row, sparse_mx.col))
    indices = torch.LongTensor(indices)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

"""
Embedding labels to one-hot form.
Input:labels: (LongTensor) class labels, sized [N,].
Outputs:(tensor) encoded labels, sized [N, #classes].
"""
def one_hot_embedding(labels):
    num_classes = labels.max().item() + 1
    y = torch.eye(num_classes) 
    return y[labels].type(torch.FloatTensor)