import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
import numpy as np
import psutil
import os   
from sklearn.model_selection import train_test_split
def khop_graphs_sparse(edge_index,num_hops):
    hops = list()
    attributes = list()
    N = edge_index.max().item() + 1
    # Create the adjacency matrix
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (N, N)).to(edge_index.device)
    # Add self loops
    I = torch.sparse_coo_tensor(torch.arange(N).unsqueeze(0).repeat(2, 1), torch.ones(N), (N, N)).to(edge_index.device)
    A = A + I
    # Degree matrix
    degrees = torch.sparse.sum(A, dim=1)
    degrees = torch.pow(degrees, -0.5)
    # Get the indices of the diagonal elements
    indices = torch.arange(N).unsqueeze(0).repeat(2, 1).to(edge_index.device)
    values = degrees.coalesce().values().to(edge_index.device)
    # Create the sparse diagonal matrix
    D_tilde = torch.sparse_coo_tensor(indices, values, (N, N)).to(edge_index.device)
    A_tilde = torch.sparse.mm(torch.sparse.mm(D_tilde, A), D_tilde)
    # Compute A_tilde^k
    A_tilde_k = A_tilde.clone().to(edge_index.device)
    hops.append(A_tilde_k.clone().coalesce().indices().to(edge_index.device))
    # Ahora ponemos los pesos de cada una de las aristas
    #attributes.append(A_tilde_k.clone().coalesce().values().to(device))
    for i in range(num_hops - 1):
        A_tilde_k = torch.sparse.mm(A_tilde_k, A_tilde)
        # We store those indices that in similarity has a value greater than 0.5
        hops.append(indices.clone())
    return hops#, attributes   
def train(data,new_edge_index,model,train_mask,optimizer,criterion):
    model.train()
    optimizer.zero_grad()
    # Get the output of the model
    out = model(data.x, data.edge_index,new_edge_index)
    # Compute the loss
    loss = criterion(out[train_mask], data.y[train_mask])
    # Compute the accuracy
    pred = out.argmax(dim=-1)
    train_correct = pred[train_mask] == data.y[train_mask]
    acc = int(train_correct.sum()) / int(train_mask.sum())
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc

@torch.no_grad()
def val(data,new_edge_index,model,val_mask):
    model.eval()
    # Get the output of the model
    out = model(data.x, data.edge_index,new_edge_index)
    # Compute the accuracy
    pred = out.argmax(dim=-1)
    val_correct = pred[val_mask] == data.y[val_mask]
    acc = int(val_correct.sum()) / int(val_mask.sum())
    return acc

@torch.no_grad()
def test(data,new_edge_index,model,test_mask):
    model.eval()
    # Get the output of the model
    out = model(data.x, data.edge_index,new_edge_index)
    # Compute the accuracy
    pred = out.argmax(dim=-1)
    test_correct = pred[test_mask] == data.y[test_mask]
    acc = int(test_correct.sum()) / int(test_mask.sum())
    return acc

def get_semi_supervised_split(data,seed):
    #We are going to use 48% of the nodes for training, 32% for testing and 20% for validation
    #First we split the nodes in 80% for training and 20% for testing
    #Then we split the training nodes in 48% for training and 32% for validation over the 80% of the nodes
    train_split, test_split = train_test_split(np.arange(len(data.y)), test_size=0.20,random_state=seed, shuffle=True)
    train_split, val_split = train_test_split(train_split, test_size=0.40,random_state=seed, shuffle=True)

    train_mask = torch.full_like(data.y, False, dtype=bool)
    train_mask[train_split] = True

    test_mask = torch.full_like(data.y, False, dtype=bool)
    test_mask[test_split] = True

    val_mask = torch.full_like(data.y, False, dtype=bool)
    val_mask[val_split] = True

    return train_mask,test_mask,val_mask