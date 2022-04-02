import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Graph Convolution layer with only one theta
    used for SingleParam, Renormalization, FirstOrderOnly, MLP
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialize weights using the initialization described in Glorot & Bengio (2010)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, preprocessed):
        support = torch.mm(input, self.weight)
        if preprocessed != None:
            output = torch.spmm(preprocessed, support)
        else:
            output = support
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class MultiValGraphConvolution(nn.Module):
    """
    Graph Convolution layer with only multiple theta
    used for Chebyshev, FirstOrder
    """

    def __init__(self, in_features, out_features, order, bias=True):
        super(MultiValGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order
        
        self.gc = []
        for i in range(self.order):
            self.gc.append(GraphConvolution(in_features, out_features))
        self.gc = nn.ModuleList(self.gc)

    def forward(self, input, preprocessed):
        output = self.gc[0](input, preprocessed[0])
        for i in range(1, self.order):
            output = output + self.gc[i](input, preprocessed[i])
        return output
