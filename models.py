import torch.nn as nn
from layers import GraphConvolution, MultiValGraphConvolution

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