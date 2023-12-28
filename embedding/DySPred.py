import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm, GraphConv

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, bias=True, activate='N'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ouput_dim = output_dim
        self.layer_num = layer_num
        self.bias = bias
        self.activate = activate

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
            if self.activate == 'N':
                x = F.selu(x)
            return x

        for layer in range(self.layer_num):
            x = self.linear[layer](x)
            if self.activate == 'N':
                x = F.selu(x)
        return x

class CoreDiffusion(nn.Module):
    def __init__(self, input_dim, output_dim, core_num=1, bias=True, rnn_type='LSTM'):
        super(CoreDiffusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.core_num = core_num
        self.bias = bias
        self.rnn_type = rnn_type

        self.linear = nn.Linear(input_dim, output_dim)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, adj_list):
        hx_list = []
        for i, adj in enumerate(adj_list):
            res = torch.sparse.mm(adj, x)
            hx_list.append(res)
        hx_list = [F.selu(res) for res in hx_list]

        hx = torch.stack(hx_list, dim=0).transpose(0, 1)  # [batch_size, core_num, input_dim]
        output, _ = self.rnn(hx)
        output = output.sum(dim=1)
        output = self.norm(output)
        return output

class CDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, diffusion_num, bias=True, rnn_type='GRU'):
        super(CDN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.diffusion_num = diffusion_num
        self.bias = bias
        self.rnn_type = rnn_type
        if diffusion_num == 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreDiffusion(input_dim, output_dim, bias=bias, rnn_type=rnn_type))
        else:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreDiffusion(input_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            for i in range(diffusion_num - 2):
                self.diffusion_list.append(CoreDiffusion(hidden_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            self.diffusion_list.append(CoreDiffusion(hidden_dim, output_dim, bias=bias, rnn_type=rnn_type))

    # adj_list: k-core subgraph adj list
    def forward(self, x, adj_list):
        for i in range(self.diffusion_num):
            x = self.diffusion_list[i](x, adj_list)
        return x


class DySPred(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, trans_num, diffusion_num, duration, bias=True, rnn_type='GRU', trans_activate_type='N'):
        super(DySPred, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.trans_activate_type = trans_activate_type

        self.duration = duration
        self.trans_num = trans_num
        self.diffusion_num = diffusion_num
        self.bias = bias

        self.mlp_list = nn.ModuleList()
        self.duffision_list = nn.ModuleList()

        for i in range(self.duration):
            self.mlp_list.append(MLP(input_dim, hidden_dim, hidden_dim, trans_num, bias=bias, activate=trans_activate_type))
            self.duffision_list.append(CDN(hidden_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type))

        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            self.rnn = nn.GRU(output_dim, output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x_list, adj_list):
        time_num = len(x_list)
        hx_list, trans_list = [], []
        for i in range(time_num):
            x = self.mlp_list[i](x_list[i])
            trans_list.append(x)
            x = self.duffision_list[i](x, adj_list[i])
            hx_list.append(x)
        hx = torch.stack(hx_list).transpose(0, 1)
        out, _ = self.rnn(hx)
        out = self.norm(out).transpose(0, 1)
        return out

class CoreGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, core_num=1, bias=True, rnn_type='LSTM'):
        super(CoreGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.core_num = core_num
        self.bias = bias
        self.rnn_type = rnn_type

        self.conv1 = GraphConv(self.input_dim, self.hidden_dim, norm='both', weight=True, bias=True)
        self.conv2 = GraphConv(self.hidden_dim, self.output_dim, norm='both', weight=True, bias=True)

        self.linear = nn.Linear(input_dim, output_dim)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=output_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=output_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, adj_list):
        adj = adj_list[-1]
        g = dgl.heterograph({('_U', '_E', '_V'): (adj._indices()[0], adj._indices()[1])})

        edge_weight = adj._values()
        norm = EdgeWeightNorm(norm='both')
        norm_edge_weight = norm(g, edge_weight)

        res = self.conv1(g, x.to_dense(), edge_weight=norm_edge_weight)
        res = self.conv2(g, F.selu(res), edge_weight=norm_edge_weight)

        res = F.selu(res)

        return res

class GCN_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, diffusion_num, bias=True, rnn_type='GRU'):
        super(GCN_Layer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.diffusion_num = diffusion_num
        self.bias = bias
        self.rnn_type = rnn_type
        if diffusion_num == 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreGCN(input_dim, hidden_dim, output_dim, bias=bias, rnn_type=rnn_type))
        else:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(CoreGCN(input_dim, hidden_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            for i in range(diffusion_num - 2):
                self.diffusion_list.append(CoreGCN(input_dim, hidden_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            self.diffusion_list.append(CoreGCN(input_dim, hidden_dim, output_dim, bias=bias, rnn_type=rnn_type))

    # adj_list: k-core subgraph adj list
    def forward(self, x, adj_list):
        for i in range(self.diffusion_num):
            x = self.diffusion_list[i](x, adj_list)
        return x


class DySGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, trans_num, diffusion_num, duration, bias=True, rnn_type='GRU', trans_activate_type='N'):
        super(DySGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.trans_activate_type = trans_activate_type

        self.duration = duration
        self.trans_num = trans_num
        self.diffusion_num = diffusion_num
        self.bias = bias

        self.duffision_list = nn.ModuleList()

        for i in range(self.duration):
            self.duffision_list.append(GCN_Layer(input_dim, hidden_dim, output_dim, diffusion_num, rnn_type=rnn_type))

        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            self.rnn = nn.GRU(output_dim, output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x_list, adj_list):
        time_num = len(x_list)
        hx_list = []
        for i in range(time_num):
            x = self.duffision_list[i](x_list[i], adj_list[i])
            hx_list.append(x)
        hx = torch.stack(hx_list).transpose(0, 1)
        out, _ = self.rnn(hx)
        out = self.norm(out).transpose(0, 1)
        return out

class DySPred_Without_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, trans_num, diffusion_num, bias=True, rnn_type='GRU', trans_activate_type='N'):
        super(DySPred_Without_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        self.trans_activate_type = trans_activate_type

        self.trans_num = trans_num
        self.diffusion_num = diffusion_num
        self.bias = bias

        self.mlp_list = MLP(input_dim, hidden_dim, hidden_dim, trans_num, bias=bias, activate=trans_activate_type)
        self.duffision_list = CDN(hidden_dim, output_dim, output_dim, diffusion_num, rnn_type=rnn_type)

        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            self.rnn = nn.GRU(output_dim, output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x_list, adj_list):

        hx_list = self.duffision_list(self.mlp_list(x_list), adj_list)
        out = self.norm(hx_list)
        return out