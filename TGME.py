# Created by Karla Paniagua
# August 2025
# TG-ME training

from anndata import AnnData
import scanpy as sc
from numpy.random import default_rng
import matplotlib.pyplot as plt
import anndata
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from pathlib import Path
from TGME_norm import run
import os
import math
import anndata
import numpy as np 
import scanpy as sc
import pandas as pd 
from PIL import Image
from pathlib import Path
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder,OneHotEncoder
import pickle as pl
from sklearn import metrics
import os,copy,sys
import random
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
from sklearn.metrics import accuracy_score,f1_score

import torch
import torch.nn
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable 
import torchvision.transforms as transforms
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
import torch.optim as optim
from torch.nn import Transformer
import argparse
from torch.utils.checkpoint import checkpoint as cp

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import Sequential, BatchNorm
from typing import Callable, Iterable, Union, Tuple, Optional
import logging
from tqdm import tqdm
import time

import scanpy.external as sce
import anndata
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import distance

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Union, Callable

from utils_func import *
from adj import graph, combine_graph_dict

from augment import augment_adata
import gc
import time
import torch.nn as nn
import torch.distributed as dist


def splitdata(num_classes,test_size,val_size,x,y_label,graph_dict):
    # Initialize lists for storing graph data
    X_train_graph, X_val_graph, X_test_graph = [], [], []

    # Initialize lists to store indices for splitting
    train_indices, val_indices, test_indices = [], [], []
    for class_label in range(0,7):
        # Get indices of data samples belonging to the current class
        class_indices = np.where(y_label == class_label)[0]

        # Split the indices into train, validation, and test
        train_idx, test_idx = train_test_split(class_indices, test_size=test_size, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=42)

        # Append the indices to the respective lists
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
        
    # Shuffle the indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # Use the shuffled indices to split the data, labels, and graph consistently
    X_train = x[train_indices]
    X_val = x[val_indices]
    X_test = x[test_indices]

    y_train = y_label[train_indices]
    y_val = y_label[val_indices]
    y_test = y_label[test_indices]

    # Initialize a new dictionary
    train_graph = {}

    # Extract 'adj_norm', 'adj_label', and 'norm_value' for the specified rows
    for key, value in graph_dict.items():
        if key == 'adj_norm':
            train_graph [key] = value[train_indices, :]
        elif key == 'adj_label':
            train_graph [key] = value[train_indices, :]
        elif key == 'norm_value':
            train_graph [key] = value

    # Initialize a new dictionary
    val_graph = {}

    # Extract 'adj_norm', 'adj_label', and 'norm_value' for the specified rows
    for key, value in graph_dict.items():
        if key == 'adj_norm':
            val_graph [key] = value[val_indices, :]
        elif key == 'adj_label':
            val_graph [key] = value[val_indices, :]
        elif key == 'norm_value':
            val_graph [key] = value

    # Initialize a new dictionary
    test_graph = {}

    # Extract 'adj_norm', 'adj_label', and 'norm_value' for the specified rows
    for key, value in graph_dict.items():
        if key == 'adj_norm':
            test_graph [key] = value[test_indices, :]
        elif key == 'adj_label':
            test_graph [key] = value[test_indices, :]
        elif key == 'norm_value':
            test_graph [key] = value
    return(X_train, y_train, X_val, y_val, X_test, y_test,train_graph,val_graph,test_graph)

def CreateDataLoaders (X_train, y_train, X_val, y_val, X_test, y_test,batch_size):
    #Inputs are the arrays of the mtx and labels splitted in train, test and validation
    
    #Convert the data and graph data to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.Tensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create Tensor datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return(train_loader,val_loader,test_loader)

class GraphDataset(Dataset):
    def __init__(self, graph_batches):
        self.graph_batches = graph_batches

    def __len__(self):
        return len(self.graph_batches)

    def __getitem__(self, idx):
        return self.graph_batches[idx]

def GraphLoader(graph, batch_size):
    num_nodes = graph['adj_norm'].size(0)
    num_batches = math.ceil(num_nodes / batch_size)

    # Calculate the sizes of all batches, including the last one
    batch_sizes = [batch_size] * (num_batches - 1) + [num_nodes - (batch_size * (num_batches - 1))]

    # Create node batches with a list of indices for each batch
    node_batches = [torch.arange(i * batch_size, i * batch_size + size) for i, size in enumerate(batch_sizes)]

    # Create batches of subgraph data
    graph_batches = []

    for nodes in node_batches:
        subgraph_adjacency = graph['adj_norm'][nodes][:, nodes]
        subgraph_adj_label = graph['adj_label'][nodes][:, nodes]
        subgraph_norm_value = graph['norm_value']

        graph_batches.append({
            'adj_norm': subgraph_adjacency,
            'adj_label': subgraph_adj_label,
            'norm_value': subgraph_norm_value,
        })

    # Create DataLoader
    graph_dataset = GraphDataset(graph_batches)
    shuffle = False  # Set to False if you want to keep the order of the data

    dataloaderG = DataLoader(graph_dataset, batch_size=None, shuffle=shuffle)
    return dataloaderG

# Start defining the multiattention layer
class multiattention(nn.Module):
    def __init__(self, batch_size, n_head, n_gene, n_feature, query_gene, mode):
        super(multiattention, self).__init__()
        self.n_head = n_head
        self.n_gene = n_gene
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.mode = mode
        self.query_gene = query_gene

        self.WQ = nn.Parameter(torch.Tensor(self.n_head, n_feature, 1), requires_grad=True)
        self.WK = nn.Parameter(torch.Tensor(self.n_head, n_feature, 1), requires_grad=True)
        self.WV = nn.Parameter(torch.Tensor(self.n_head, n_feature, 1), requires_grad=True)
        torch.nn.init.xavier_normal_(self.WQ, gain=1)
        torch.nn.init.xavier_normal_(self.WK, gain=1)
        torch.nn.init.xavier_normal_(self.WV)

        self.W_0 = nn.Parameter(torch.Tensor(self.n_head * [0.001]), requires_grad=True)

    def QK_diff(self, Q_seq, K_seq):
        QK_dif = -1 * torch.pow((Q_seq - K_seq), 2)
        return torch.nn.Softmax(dim=2)(QK_dif)

    def mask_softmax_self(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = x.shape[1]
        x = x * ((1 - torch.eye(d, d)).to(device))
        return x

    def attention(self, x, Q_seq, WK, WV):
        if self.mode == 0:
            K_seq = x * WK
            K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.n_gene)
            K_seq = K_seq.permute(0, 2, 1)
            V_seq = x * WV
            QK_product = Q_seq * K_seq
            z = torch.nn.Softmax(dim=2)(QK_product)
            z = self.mask_softmax_self(z)
            out_seq = torch.matmul(z, V_seq)
        return out_seq

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        out_h = []
        save_memory=False
        for h in range(self.n_head):
            Q_seq = x * self.WQ[h, :, :]
            Q_seq = Q_seq.expand(Q_seq.shape[0], Q_seq.shape[1], self.n_gene)
            if save_memory:
                attention_out = cp(self.attention, x, Q_seq, self.WK[h, :, :], self.WV[h, :, :])
            else:
                attention_out = self.attention(x, Q_seq, self.WK[h, :, :], self.WV[h, :, :])
            out_h.append(attention_out)
        out_seq = torch.cat(out_h, dim=2)
        out_seq = torch.matmul(out_seq, self.W_0)
        return out_seq

# Layer normalization
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, out):
        return x + self.norm(self.dropout(out))

class GCNModel(nn.Module):
    def __init__(self, input_dim,
                conv_hidden=[32,8],
                p_drop=0.1,
                dec_cluster_n=18,
                activate="relu",
                ):
        super(GCNModel, self).__init__()
        self.input_dim = input_dim
        self.alpha = 0.8
        self.conv_hidden = conv_hidden
        self.activate = activate
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n
        current_encoder_dim = self.input_dim
        
        self.conv = Sequential('x, edge_index', [
                        (GraphConv(input_dim, conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
        self.conv_mean = Sequential('x, edge_index', [
                        (GraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
        self.conv_logvar = Sequential('x, edge_index', [
                        (GraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
        #self.dc = InnerProductDecoder(p_drop)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.input_dim+self.conv_hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        
    def encode(
        self, 
        x, 
        adj,
        ):
        conv_x = self.conv(x, adj)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj)
    
    def reparameterize(
        self, 
        mu, 
        logvar,
        ):
        #if self.training:
            #print('yes')
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        #else:
            #print(mu)
         #   return mu
        
    def decoder(self, z):
        #print(z)
        decoder=nn.Sequential(nn.Linear(self.input_dim+self.conv_hidden[-1],128),
                             nn.ReLU(inplace=True),nn.Linear(128,self.input_dim),)
        decoder=decoder.to(z.device)
        decoded_output=decoder(z)
        return decoded_output
    
    def forward(
        self, 
        x, 
        adj
        ):
        mu, logvar = self.encode(x, adj)
        if self.training:
            gnn_z = self.reparameterize(mu, logvar)
        else:
            gnn_z=mu
            #print(mu)
        z = torch.cat((x, gnn_z), 1)
        #print(z)
        de_feat = self.decoder(z)
        
        #
        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return z, mu, logvar, de_feat, q, gnn_z

# Define your Transformer-based model
class TransformerModel(nn.Module):
    def __init__(self, batch_size, n_head, n_gene, n_feature, n_class, query_gene, d_ff, dropout_rate, mode):
        super(TransformerModel, self).__init__()
        self.n_head = n_head
        self.n_gene = n_gene
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.n_class = n_class
        self.query_gene = query_gene
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.mode = mode
        self.multiattention1 = multiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, self.query_gene, self.mode)
        self.multiattention2 = multiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, self.query_gene, self.mode)
        self.multiattention3 = multiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, self.query_gene, self.mode)
        self.fc = nn.Linear(self.n_gene, self.n_class)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=1)
        self.ffn1 = nn.Linear(self.n_gene, self.d_ff)
        self.ffn2 = nn.Linear(self.d_ff, self.n_gene)
        self.dropout = nn.Dropout(dropout_rate)
        self.sublayer = ResidualConnection(self.n_gene, dropout_rate)
        self.output_projection=nn.Linear(n_gene, 128)

    def feedforward(self, x):
        out = F.relu(self.ffn1(x))
        out = self.ffn2(self.dropout(out))
        return out

    def forward(self, gcn_embeddings):  # Pass GCN embeddings as input
        # Integrate GCN embeddings into the input
        #print(x.shape)
        #print(gcn_embeddings.shape)
        x = gcn_embeddings
        act_fun=None
        out_attn = self.multiattention1(x)
        out_attn_1 = self.sublayer(x, out_attn)
        out_attn_2 = self.multiattention2(out_attn_1)
        out_attn_2 = self.sublayer(out_attn_1, out_attn_2)
        out_attn_3 = self.multiattention3(out_attn_2)
        out_attn_3 = self.sublayer(out_attn_2, out_attn_3)
        if act_fun == 'relu':
            out_attn_3 = F.relu(out_attn_3)
        if act_fun == 'leakyrelu':
            m = torch.nn.LeakyReLU(0.1)
            out_attn_3 = m(out_attn_3)
        if act_fun == 'gelu':
            m = torch.nn.GELU()
            out_attn_3 = m(out_attn_3)
        #out_attn_3=self.output_projection(out_attn_3)
        y_pred = self.fc(out_attn_3)
        y_pred = F.log_softmax(y_pred, dim=1)

        return out_attn_3,y_pred

# Define integrated model TGME
class TGME(nn.Module):
    def __init__(self, batch_size, n_head, n_gene,num_classes, d_ff, dropout_rate, mode, input_dim, conv_hidden, p_drop,
                dec_cluster_n,activate):
        super(TGME, self).__init__()

        # GCN layers
        #gcn_model = GCNModel(num_features, hidden_channels, num_classes)
   
        # Transformer layers
        self.transformer = TransformerModel(batch_size, n_head, n_gene, n_gene,num_classes, n_gene, d_ff, dropout_rate, mode)
        self.gcn_model= GCNModel(input_dim, conv_hidden=[32,32],p_drop=0.1, dec_cluster_n=15, activate="relu")
        #self.prediction_layer= nn.Linear(hidden_channels, dec_cluster_n)
        #self.prediction_layer=nn.Linear(input_dim+conv_hidden[-1],num_classes)
        
    def forward(self, x, edge_index):
        # Forward pass through 
        embeddings,y_pred= self.transformer(x)
        z, mu, logvar, de_feat, q, gnn_z= self.gcn_model(embeddings,edge_index)
        #y_pred=self.prediction_layer(z)
        #y_pred=F.log_softmax(y_pred,dim=1)
        #predictions=self.prediction_layer(gcn_output)
        #y_pred = F.log_softmax(predictions, dim=1)
        #print(gcn_output[0].shape)
        # Forward pass through Transformer
          # Pass GCN output as input

        return y_pred,de_feat,embeddings,z,gnn_z

def train_TGME(model,lr,weight_decay,num_epochs,verbose,verbose_interval,train_loader,val_loader,dataloader_trainG,dataloader_valG,path_save):
    # model is the defined model
    # lr is the learning rate
    # weight_decay is the weight decay
    # num_epochs number of epochs to train the model for
    # verbose True/False
    # verbose_interval increase the verbose every X epochs
    # train_loader data loader for training data
    # val_loader data loader for validation data
    # dataloader_trainG data loader for trainig graph
    # dataloader_valG data loader for validation graph
    # path_save where to save the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('Training the model...')
    # Define the progress bar for training
    train_bar = tqdm(total=len(train_loader), desc='Training', position=0, leave=True)

    # Define the progress bar for validation
    val_bar = tqdm(total=len(val_loader), desc='Validation', position=0, leave=True)

    best_val_loss = float('inf')  # Initialize the best validation loss
    patience = 10  # Number of epochs with no improvement to wait before early stopping
    no_improvement_count = 0  # Initialize the no improvement counter
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()  # Set the model to training mode
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (batch_X, batch_G) in enumerate(zip(train_loader, dataloader_trainG)):
            inputs = batch_X[0].to(device)   # Adjust for your data format
            labels = batch_X[1].to(device)   # Adjust for your data format
            labels = labels.view(-1)
            adj_label = batch_G['adj_label'].to(device)
            adj = batch_G['adj_norm'].to(device)
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass through the model
            deeptme_output,de_feat,z,e,gnn= model(inputs, adj)  # Pass edge_index for GCN
            #print(gnn)
            #print(embeddings.shape)
            #print(de_feat.shape)

            # Calculate loss (you may need to adapt the loss function)
            loss1= F.nll_loss(deeptme_output, labels)
            loss2=F.mse_loss(z,de_feat)
            class_weight=0.8
            auto_weight=0.5
            loss=class_weight*loss1+auto_weight*loss2
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()
            _, predicted = torch.max(deeptme_output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            inputs.detach()
            labels.detach()
            adj.detach()
            adj_label.detach()

            train_bar.update(1)  # Update the training progress bar

        # Calculate accuracy and average loss for the epoch
        accuracy = 100.0 * correct_predictions / total_samples
        average_loss = total_loss / len(train_loader)

        # Print verbose output for the epoch
        #print(f"Epoch [{epoch + 1}/{num_epochs}]")
        #print(f"Train Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

        train_bar.n = 0  # Reset the training progress bar
        train_bar.last_print_n = 0
        train_bar.refresh()

        # Validation
    
        total_val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            model.eval()
            for i, (val_batch, batch_valG) in enumerate(zip(val_loader, dataloader_valG)):
                inputs = val_batch[0].to(device)   # Adjust for your data format
                labels = val_batch[1].to(device)   # Adjust for your data format
                labels = labels.view(-1)
                adj_label = batch_valG['adj_label'].to(device)
                adj = batch_valG['adj_norm'].to(device)
                #print(inputs.shape,adj)
                #print(adj)

                # Forward pass through the model for validation
                val_deeptme_output,vde_feat,zv,emb,gnn_z = model(inputs, adj)   # Pass edge_index for GCN
                #print(gnn_z)
                # Calculate validation loss (use the same loss function)
                #val_loss1 = F.nll_loss(val_deeptme_output, labels)
                #val_loss2=F.mse_loss(zv,vde_feat)
                #val_class_weight=1.0
                #val_auto_weight=1.0
                #val_loss=class_weight*loss1+auto_weight*loss2
                vloss1= F.nll_loss(val_deeptme_output, labels)
                vloss2=F.mse_loss(zv,vde_feat)
                vclass_weight=0.8
                vauto_weight=0.5
                val_loss=vclass_weight*vloss1+vauto_weight*vloss2
                

                total_val_loss += val_loss.item()
                _, val_predicted = torch.max(val_deeptme_output, 1)
                correct_val_predictions += (val_predicted == labels).sum().item()
                total_val_samples += labels.size(0)

                inputs.detach()
                labels.detach()
                adj.detach()
                adj_label.detach()

                val_bar.update(1)  # Update the validation progress bar

        # Calculate validation accuracy and average loss for the epoch
        val_accuracy = 100.0 * correct_val_predictions / total_val_samples
        val_average_loss = total_val_loss / len(val_loader)

        # Print validation results for the epoch
        #print(f"Validation Loss: {val_average_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Check for early stopping
        if val_average_loss < best_val_loss:
            best_val_loss = val_average_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping after {patience} epochs of no improvement.")
            break  # Stop training if no improvement for 'patience' epochs

        val_bar.n = 0  # Reset the validation progress bar
        val_bar.last_print_n = 0
        val_bar.refresh()
        
        # Print epoch duration
        epoch_end_time = time.time()
        #print(f"Epoch {epoch + 1} duration: {epoch_end_time - epoch_start_time:.2f} seconds")

    end_time = time.time()
    # Close the progress bars
    train_bar.close()
    val_bar.close()
    print(f"Train Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Validation Loss: {val_average_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        

    #torch.save(model, '/work/jfn045/HRST/FC/' + str(n_head) +'_epoch'+str(num_epochs)+'.model')
    print('Done!')
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    print('Saving model...')
    torch.save(model.state_dict(), path_save +'TGME-model.pt')
    print('Done!')
    return model


def AllDataLoader (X_train, y_train, batch_size):
    #Inputs are the arrays of the mtx and labels splitted in train, test and validation
    
    #Convert the data and graph data to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    
    # Create Tensor datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    
    return(train_loader)

def extract_embeddings(pretrained_model,X_dataloader, dataloader_G):
    # model is the pre-trained model
    # X_dataloader is the data loader for all data
    # dataloader_G is the data loader of the graph
    # Initialize empty lists to store true labels and predicted labels
    # Initialize empty lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []
    embeddings_final=[]

    # Set the model to evaluation mode
    pretrained_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a tqdm progress bar
    accuracy_bar = tqdm(total=len(X_dataloader), desc='Calculating Accuracy', position=0, leave=True)

    with torch.no_grad():
        for batch_X, batch_G in zip(X_dataloader, dataloader_G):
            inputs = batch_X[0].to(device)   # Adjust for your data format
            adj = batch_G['adj_norm'].to(device)

            # Pass the gene expression data and adjacency matrix to GCN
            #gcn_output = pretrained_model.gcn_model(inputs, adj)


            # Forward pass through the model to obtain predictions and embeddings
            predictions,de_feat,_,embeddings,_ = pretrained_model(inputs, adj)

            # Extract true labels (replace with your actual labels)
            true_labels.extend(batch_X[1].tolist())  # Adjust for your data format

            # Convert predictions to labels (assuming predictions are log probabilities)
            predicted_labels.extend(predictions.argmax(dim=1).tolist())
            
            embeddings_final.append(embeddings)

            # Update the progress bar
            accuracy_bar.update(1)

    # Close the progress bar
    accuracy_bar.close()

    all_embeddings = torch.cat(embeddings_final, dim=0)
    all_embeddings=all_embeddings.to('cpu')
    embeddings=pd.DataFrame(all_embeddings.numpy())
    return embeddings

def TGME_clustering(embeddings,adata, n_neighbors,resolution):
    adata.obsm['TG-ME']=np.array(embeddings)
    sc.pp.neighbors(adata, use_rep='TG-ME', n_neighbors=n_neighbors)
    #sc.tl.leiden(adata, key_added="TG-ME_domain",resolution,use_weights=True,)
    print('Computing clustering...')
    sc.tl.leiden(adata, key_added="TG-ME_domain",resolution=resolution,use_weights=True,obsp='connectivities')
    ######### Strengthen the distribution of points in the model
    print('Evaluating clustering...')
    adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
    print('Refining clustering...')
    refined_pred= refine(sample_id=adata.obs.index.tolist(), 
    pred=adata.obs["TG-ME_domain"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["TG-ME_refine_domain"]= refined_pred
    print('Done')
    return adata
