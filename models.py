import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, MultiValGraphConvolution


class GCN(nn.Module):
    def __init__(self, input, hidden, output, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input, hidden)
        self.gc2 = GraphConvolution(hidden, output)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class MultiValGCN(nn.Module):
    def __init__(self, input, hidden, output, order, dropout):
        super(MultiValGCN, self).__init__()

        self.gc1 = MultiValGraphConvolution(input, hidden, order)
        self.gc2 = MultiValGraphConvolution(hidden, output, order)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)