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
