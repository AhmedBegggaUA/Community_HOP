import argparse
import os
import torch
import numpy as np
import pandas as pd
import time
from torch_geometric.datasets import Planetoid,WebKB,Actor,WikipediaNetwork, LINKXDataset, Amazon, Coauthor
from torch_geometric.transforms import *
import torch_geometric.transforms as T
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from torch_geometric.utils import to_networkx
from models import *
from utils import *
import warnings
import networkx as nx
import numpy as np
import pickle
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.seed import seed_everything
#seed_everything(1234)ç
import sys
# Fix the seed of the system
os.environ['PYTHONHASHSEED'] = str(1234)
import random
seed = 1234
warnings.filterwarnings("ignore")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# Si estás utilizando CUDA
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # Para multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
################### Arguments parameters ###################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="cora",
    choices=["texas","wisconsin","cornell",
             "actor",
             "squirrel","chameleon",
             "cora","citeseer","pubmed",
             "penn94",
             "computers","photo",
             "cs","physics"],
    help="You can choose between texas, wisconsin, actor, cornell, squirrel, chamaleon, cora, citeseer, pubmed and penn94",
    )
parser.add_argument(
    "--cuda",
    default="cuda:0",
    choices=["cuda:0","cuda:1","cpu"],
    help="You can choose between cuda:0, cuda:1, cpu",
    )
parser.add_argument(
        "--hidden_channels", type=int, default=64, help="Hidden channels for the unsupervised model"
    )
parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate"
    )
parser.add_argument(
        "--lr", type=float, default=0.01, help="Outer learning rate of model"
    )
parser.add_argument(
        "--wd", type=float, default=5e-4, help="Outer weight decay rate of model"
    )
parser.add_argument(
        "--epochs", type=int, default=2000, help="Epochs for the model"
    )
parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers for the model"
    )
parser.add_argument(
        "--model", type=str, default="GCN", help="Model to use"
    )
parser.add_argument(
        "--community", type=bool, default=True, help="Receptive field"
    )
parser.add_argument(
        "--k", type=int, default=6, help="Number of walks"
    )
parser.add_argument(
        "--hops", type=int, default=4, help="Hops for the model"
    )
args = parser.parse_args()
args.cuda = 'cpu' if not torch.cuda.is_available() else 'cuda:0'

################### Importing the dataset ###################################

if args.dataset in ["wisconsin","cornell","texas"]:
    transform = T.Compose([T.LargestConnectedComponents()])
    dataset = WebKB(root='./data',name=args.dataset,transform=transform)
    data = dataset[0]

elif args.dataset == "actor":
    transform = T.Compose([ T.ToUndirected(),T.LargestConnectedComponents()])
    dataset  = Actor(root='./data', transform=transform)
    dataset.name = "film"
    data = dataset[0]

elif args.dataset in ["squirrel","chameleon"]:
    transform = T.Compose([T.ToUndirected(),T.LargestConnectedComponents()])
    dataset = WikipediaNetwork(root='./data',name=args.dataset,transform=transform)
    data = dataset[0]    

elif args.dataset in ["pubmed","cora","citeseer"]:
    transform = T.Compose([T.LargestConnectedComponents()])
    dataset = Planetoid(root='./data',name=args.dataset,transform= transform)
    data = dataset[0]
elif args.dataset == "penn94":
    transform = T.Compose([T.ToUndirected(),T.LargestConnectedComponents()])
    dataset = LINKXDataset(root='./data',name=args.dataset,transform= transform)
    data = dataset[0]
elif args.dataset in ["computers","photo"]:
    transform = T.Compose([T.ToUndirected(),T.LargestConnectedComponents()])
    dataset = Amazon(root='./data',name=args.dataset,transform= transform)
    data = dataset[0]
elif args.dataset in ["cs","physics"]:
    transform = T.Compose([T.ToUndirected(),T.LargestConnectedComponents()])
    dataset = Coauthor(root='./data',name=args.dataset,transform= transform)
    data = dataset[0]
else:
    raise ValueError("Dataset not found")
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print()
print(data) 
print('===========================================================================================================')
if args.community:
    print('===========================================================================================================')
    print('Using Community HOP')
    print('===========================================================================================================')
    G = to_networkx(data, to_undirected=True)
    L = nx.normalized_laplacian_matrix(G).toarray()
    # We sort the eigenvectors
    # Calcular los primeros k vectores propios (omitiendo el primero)
    k = int(args.k) # Número de clusters
    _, eigenvectors = eigsh(L, k=k+1, which='SM')
    # Aplicar k-means en los vectores propios para obtener las comunidades
    kmeans = KMeans(n_clusters=k, random_state=1234).fit(eigenvectors[:, 1:])
    clusters = kmeans.labels_
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    # Now we remove those edges that are connecting nodes from different communities
    mask = clusters[edge_index[0]] == clusters[edge_index[1]]
    new_edge_index = edge_index[:, mask]
    print("The diameter of the graph is: ",nx.diameter(G))
    # G_news = []
    # # For each community we create a new graph
    # for i in range(k):
    #     mask = clusters == i
    #     edge_index = new_edge_index
    #     edge_index = edge_index[:, (mask[edge_index[0]] & mask[edge_index[1]])]
    #     G_new = nx.Graph()
    #     G_new.add_edges_from(edge_index.t().tolist())
    #     G_news.append(G_new)
    # print("The new graphs have the following diameters: ",[nx.diameter(G_new) for G_new in G_news])
    print('===========================================================================================================')
    print('Computing the hops')
    print('===========================================================================================================')
    new_edge_indexs = khop_graphs_sparse(new_edge_index,args.hops)
    
################### CUDA ###################################
device = torch.device(args.cuda)
data = data.to(device)   
new_edge_indexs = [x.to(device) for x in new_edge_indexs]
print("Device: ",device)
results = []
for i in range(10):
    # Time per split
    #start = time.time()
    #train_mask,test_mask,val_mask = get_semi_supervised_split(data.cpu(), i)
    with open('splits/'+dataset.name+'_split_0.6_0.2_'+str(i)+'.npz', 'rb') as f:
                splits = np.load(f)
                train_mask = torch.tensor(splits['train_mask']).to(device)
                val_mask = torch.tensor(splits['val_mask']).to(device)
                test_mask = torch.tensor(splits['test_mask']).to(device)        
    train_mask = train_mask.to(device)
    # We make sure the masks have the same number of nodes as the data
    train_mask = train_mask[:data.x.shape[0]]
    val_mask = val_mask.to(device)
    val_mask = val_mask[:data.x.shape[0]]
    test_mask = test_mask.to(device)
    test_mask = test_mask[:data.x.shape[0]]
    data = data.to(device)
    print("Train mask shape: ",train_mask.shape)
    print("Val mask shape: ",val_mask.shape)
    print("Test mask shape: ",test_mask.shape)
    print("Data x shape: ",data.x.shape)
    print('===========================================================================================================')
    print('Split: ',i)
    print('===========================================================================================================')
    model = CommunityHOP(in_channels=data.x.shape[1],
                hidden_channels=args.hidden_channels,
                out_channels=data.y.max().item()+1,
                dropout=args.dropout,
                hops=args.hops
                ).to(device)
    # model = MLP(in_channels=data.x.shape[1],
    #             hidden_channels=args.hidden_channels,
    #             out_channels=data.y.max().item()+1
    #             ).to(device)
    # model = GCN(in_channels=data.x.shape[1],
    #             hidden_channels=args.hidden_channels,
    #             out_channels=data.y.max().item()+1,
    #             hops=args.hops
    #             ).to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    test_acc = 0
    patience = 0
    best_test = 0
    for epoch in range(args.epochs):
        loss,acc_train = train(data,new_edge_indexs,model,train_mask,optimizer,criterion)
        acc_val = val(data,new_edge_indexs,model,val_mask)
        acc_test = test(data,new_edge_indexs,model,test_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}')
        if acc_val >= test_acc:
            test_acc = acc_val
            if best_test < acc_test:
                best_test = acc_test
        # if test_acc > acc_test:
        #     patience += 1
        # else:
        #     patience = 0
        # if patience == 200:
        #     break
    #end = time.time()
    del model
    print('===========================================================================================================')
    print('Test Accuracy: ',best_test)#,'Time: ',end-start)
    print('===========================================================================================================')
    results.append(best_test)
print('===========================================================================================================')
print('Report: ',np.mean(results)*100,'+-',np.std(results)*100)
print('===========================================================================================================')
print(' Configuration: ',args)
print('===========================================================================================================')

# Now we check if it is created a csv with the configuration and the results
if os.path.isfile('results.csv'):
    # If the file exists, then we append the configuration and the results
    # The columns are: dataset, model, hidden_channels, lr, epochs, num_centers, AUC, AP
    res = pd.read_csv('results.csv')
    # Check if the configuration is already in the csv
    
    res = pd.concat([res, pd.DataFrame({'model': args.model, 'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'num_layers': args.num_layers, 'wd': args.wd, 'k': args.k, 'hops': args.hops, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, index=[0])], ignore_index=True)
    #res.to_csv('results.csv', index=False)
    res.to_csv('results.csv', index=False)
else:
    # If the file does not exist, then we create it and append the configuration and the results
    res = pd.DataFrame(columns=['model','dataset', 'hidden_channels', 'lr', 'dropout', 'epochs', 'num_layers', 'wd', 'k', 'hops', 'Accuracy', 'std'])
    #res = res.append({'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'num_layers': args.num_layers, 'n_layers': args.n_layers, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, ignore_index=True)
    res = pd.concat([res, pd.DataFrame({'model': args.model, 'dataset': args.dataset, 'hidden_channels': args.hidden_channels, 'lr': args.lr, 'dropout': args.dropout, 'epochs': args.epochs, 'num_layers': args.num_layers, 'wd': args.wd, 'k': args.k, 'hops': args.hops, 'Accuracy': np.round(np.mean(np.array(results))*100,2), 'std': np.round(np.std(np.array(results))*100,2)}, index=[0])], ignore_index=True)
    res.to_csv('results.csv', index=False)
