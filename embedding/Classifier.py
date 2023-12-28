import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, bias=True, activate_type='N'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ouput_dim = output_dim
        self.layer_num = layer_num
        self.bias = bias
        self.activate_type = activate_type

        if layer_num == 1:
            self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            self.linear = torch.nn.ModuleList()
            self.linear.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            for layer in range(layer_num - 2):
                self.linear.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.linear.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, x):
        if self.layer_num == 1:
            x = self.linear(x)
            if self.activate_type == 'N':
                x = F.selu(x)
            return x

        for layer in range(self.layer_num):
            x = self.linear[layer](x)
            if self.activate_type == 'N':
                x = F.selu(x)
        return x

# MLP classifier
class MLPClassifier(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    layer_num: int
    duration: int
    bias: bool
    activate_type: str

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, duration, bias=True, activate_type='N'):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.duration = duration
        self.bias = bias
        self.activate_type = activate_type

        self.mlp_list = nn.ModuleList()
        for i in range(self.duration):
            self.mlp_list.append(MLP(input_dim, hidden_dim, output_dim, layer_num, bias=bias, activate_type=activate_type))

    def forward(self, x, batch_indices=None):
        if isinstance(x, list) or len(x.shape) == 3:  # list or 3D tensor(GCRN, CTGCN output)
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                embedding_mat = x[i][batch_indices] if batch_indices is not None else x[i]
                out = self.mlp_list[i](embedding_mat)
                output_list.append(out)
            return output_list
        else:
            embedding_mat = x[batch_indices] if batch_indices is not None else x
            return self.mlp_list[0](embedding_mat)

# This class supports inner product edge features!
class InnerProduct(nn.Module):
    reduce: bool

    def __init__(self, reduce=True):
        super(InnerProduct, self).__init__()
        self.reduce = reduce

    def forward(self, x, edge_index):
        if isinstance(x, list) or len(x.shape) == 3:  # list or 3D tensor(GCRN, CTGCN output)
            timestamp_num = len(x)
            output_list = []
            for i in range(timestamp_num):
                embedding_mat = x[i]
                edge_mat = edge_index[i]
                output_list.append(self.inner_product(embedding_mat, edge_mat))
            return output_list
        else:
            return self.inner_product(x, edge_index)

    def inner_product(self, x, edge_index):
        embedding_i = x[edge_index[:, 0].long()]
        embedding_j = x[edge_index[:, 1].long()]
        if self.reduce:
            return torch.sum(embedding_i * embedding_j, dim=1)
        return embedding_i * embedding_j


class EdgeClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, duration, bias=True, activate_type='N'):
        super(EdgeClassifier, self).__init__()
        self.conv = InnerProduct(reduce=False)
        self.classifier = MLPClassifier(input_dim, hidden_dim, output_dim, layer_num, duration, bias=bias, activate_type=activate_type)

    def forward(self, x, edge_index):
        conv = self.conv(x, edge_index)
        return self.classifier(conv)
