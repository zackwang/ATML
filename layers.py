import torch
import torch.nn as nn

"""
Graph Convolution layer with only one theta
used for SingleParam, Renormalization, FirstOrderOnly, MLP
"""
class GraphConvolution(nn.Module):
    def __init__(self, input, output, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input, output))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weights using the initialization described in Glorot & Bengio (2010)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, preprocessed):
        x = torch.mm(x, self.weight)
        if preprocessed is not None:
            x = torch.spmm(preprocessed, x)
        
        if self.bias is not None:
            x = x + self.bias
        
        return x

"""
Graph Convolution layer with multiple theta
used for Chebyshev, FirstOrder
"""
class MultiValGraphConvolution(nn.Module):
    def __init__(self, input, output, order, bias=True):
        super(MultiValGraphConvolution, self).__init__()
        self.order = order
        
        self.gc = []
        for i in range(self.order):
            self.gc.append(GraphConvolution(input, output))
        self.gc = nn.ModuleList(self.gc)

    def forward(self, x, preprocessed):
        output = self.gc[0](x, preprocessed[0])
        for i in range(1, self.order):
            output = output + self.gc[i](x, preprocessed[i])
        return output

"""
GCN-LPA Layer
"""

class GCNLPAConv(nn.Module):
    def __init__(self, input, output, adj, lpa_iter, bias=True):
        super(GCNLPAConv, self).__init__()
        self.adj = adj
        self.lpa_iter = lpa_iter
        self.weight = nn.Parameter(torch.FloatTensor(input, output))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        # initialize weights using the initialization described in Glorot & Bengio (2010)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, y, adj_mask):
        # normalized adjancy matrix
        mask = torch.where(self.adj>0, adj_mask, -9e9*torch.ones_like(adj_mask))
        adj = torch.softmax(mask, dim=1)
        # W * x
        support = torch.mm(x, self.weight) 
        # output = D^-1 * A' * X * W
        output = torch.mm(adj, support)
        # y' = D^-1 * A' * y
        
        # LPA regularization
        y_hat = y 
        for _ in range(self.lpa_iter):
            y_hat = torch.mm(adj, y_hat)

        if self.bias is not None:
            return output + self.bias, y_hat
        else:
            return output, y_hat

"""
GAT Layer
"""
class GraphAttentionLayer(nn.Module):
    def __init__(self, input, output, dropout, alpha, concat=True, bias=False):
        super(GraphAttentionLayer, self).__init__()
        self.output = output
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(input, output)))
        self.a = nn.Parameter(torch.empty(size=(2*output, 1)))

        self.elu = nn.ELU()
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) 
        a_input = self._prepare_attentional_mechanism_input(Wh) 
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        
        attention = torch.where(adj > 0, e, zero_vec)
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        h_prime = torch.matmul(adj, Wh)      
        h_prime = torch.matmul(attention, h_prime)

        if self.concat:
            return self.elu(h_prime)
        else:
            return h_prime
        
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.output)