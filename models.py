import torch
import torch.nn as nn
from layers import GraphConvolution, MultiValGraphConvolution, GCNLPAConv, GraphAttentionLayer

class GCN(nn.Module):
    def __init__(self, input, hidden, output, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input, hidden)
        self.dropout1 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.gc2 = GraphConvolution(hidden, output)
        self.dropout2 = nn.Dropout(p=dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, preprocessed):
        x = self.gc1(x, preprocessed)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.gc2(x, preprocessed)
        x = self.dropout2(x)
        x = self.softmax(x)
        return x


class MultiValGCN(nn.Module):
    def __init__(self, input, hidden, output, order, dropout):
        super(MultiValGCN, self).__init__()

        self.gc1 = MultiValGraphConvolution(input, hidden, order)
        self.dropout1 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.gc2 = MultiValGraphConvolution(hidden, output, order)
        self.dropout2 = nn.Dropout(p=dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, preprocessed):
        x = self.gc1(x, preprocessed)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.gc2(x, preprocessed)
        x = self.dropout2(x)
        x = self.softmax(x)
        return x

class GCNLPA(nn.Module):
    def __init__(self, input, hidden, output, adj, dropout, layer_num, lpa_iter):
        super(GCNLPA, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLPAConv(input, hidden, adj, lpa_iter))

        for _ in range(layer_num-2):
            self.layers.append(GCNLPAConv(hidden, hidden, adj, lpa_iter))

        self.layers.append(GCNLPAConv(hidden, output, adj, lpa_iter))
        self.dropout = nn.Dropout(p=dropout)
        self.elu = nn.ELU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.adj_mask = nn.Parameter(adj.clone())

    def forward(self, x, y):
        y_hat = y
        for net in self.layers[:-1]:
            x = self.dropout(x)
            x, y_hat = net(x, y, self.adj_mask)
            x = self.elu(x)
            
        x = self.dropout(x)
        x, y_hat = self.layers[-1](x, y_hat, self.adj_mask)
        x = self.softmax(x)
        y_hat = self.softmax(y_hat)
            
        return x, y_hat

class GAT(nn.Module):
    def __init__(self, input, hidden, output, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.attentions = [GraphAttentionLayer(input, hidden, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hidden * nheads, output, dropout=dropout, alpha=alpha, concat=False)
        self.dropout = nn.Dropout(p=dropout)
        self.elu = nn.ELU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.dropout(x)
        x = self.elu(self.out_att(x, adj))
        x = self.softmax(x)
        return x

