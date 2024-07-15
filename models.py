import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,SAGEConv
from torch.nn import Linear
def init_weights(m):
    """Performs weight initialization.

    Args:
        m (nn.Module): PyTorch module

    """
    if (isinstance(m, torch.nn.BatchNorm2d)
            or isinstance(m, torch.nn.BatchNorm1d)):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, torch.nn.Linear):
        m.weight.data = torch.nn.init.xavier_uniform_(
            m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.zero_()
#from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
class MLP(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,seed=12345):
        super(MLP, self).__init__()
        # fix the seed
        torch.manual_seed(seed)
        # 2-layer GCN
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.loss_euclidean = 0
    def forward(self, x, edge_index,new_edge_indexs):
        # edge_index is not used
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x.log_softmax(dim=-1)#.softmax(dim=1)

class GCN(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,hops,seed=12345):
        super(GCN, self).__init__()
        # fix the seed
        torch.manual_seed(seed)
        # 2-layer GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, x, edge_index,new_edge_indexs):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=-1)#.softmax(dim=1)
class GAT(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,hops,seed=12345):
        super(GAT, self).__init__()
        # fix the seed
        torch.manual_seed(seed)
        # 2-layer GCN
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
    def forward(self, x, edge_index,new_edge_indexs):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=-1)#.softmax(dim=1)
class CommunityHOP(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,dropout,hops,seed=12345):
        super(CommunityHOP, self).__init__()
        torch.manual_seed(seed)
        # 2-layer GCN
        #Extra conv
        cached = False
        add_self_loops = True
        save_mem  = False
        self.conv_extra = GCNConv(in_channels, hidden_channels)#, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops)
        #self.bn_extra = torch.nn.BatchNorm1d(hidden_channels)
        self.conv_extra2 = GCNConv(hidden_channels, hidden_channels)#, cached=cached, normalize=not save_mem , add_self_loops=add_self_loops)       
        #self.bn_extra_2 = torch.nn.BatchNorm1d(hidden_channels)
        self.lins = torch.nn.ModuleList()
        # we add as many linear layers as hops
        self.mlp = Linear(in_channels, hidden_channels)
        for i in range(hops):
            self.lins.append(GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
        self.hops = hops
        # We add attention mechanism
        self.att = nn.Parameter(torch.ones(hops + 2))
        self.sm = nn.Softmax(dim=0)
        self.dropout_parameter = dropout
        self.classify = Linear(hidden_channels * (hops + 2), out_channels)
        # Initialize weights
        self.apply(init_weights)
    def forward(self, x, edge_index,new_edge_indexs):
        mask = self.sm(self.att)
        # Typical GCN
        mlp = self.mlp(x).relu() * mask[0]
        extra_conv = self.conv_extra(x, edge_index)
        #extra_conv = self.bn_extra(extra_conv)
        extra_conv = extra_conv.relu()
        extra_conv = F.dropout(extra_conv, p=0.5, training=self.training)
        extra_conv = self.conv_extra2(extra_conv, edge_index)
        #extra_conv = self.bn_extra_2(extra_conv)
        extra_conv = extra_conv.relu()
        extra_conv = extra_conv * mask[1] 
        # Community HOP    
        outs = list()    
        #outs.append(self.mlp(x).relu()*mask[1])
        for i, lin in enumerate(self.lins):
            out = lin(x, new_edge_indexs[i]).relu()*mask[i+1]
            outs.append(out.clone())
        outs = torch.cat(outs, dim=1)        
        # now we concat x and x_original
        z = torch.cat([outs, extra_conv,mlp],dim=1)
        z = F.dropout(z, p=self.dropout_parameter, training=self.training)
        z = self.classify(z)
        return z.log_softmax(dim=-1)