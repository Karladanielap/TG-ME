#!/usr/bin/env python
"""
# Author: ChangXu
# Created Time : Mon 23 Apr 2021 08:26:32 PM CST
# File Name: STMAP_train.py
# Description:`

"""
"""
test:
from cal_graph import graph
import scanpy as sc
from GCN import STMAP_model
from STMAP_main import train
data_path = "/home/xuchang/Project/STMAP/Human_breast/output/Breast_data/STMAP_Breast_15.h5ad"
adata = sc.read(data_path)
graph_dict = graph(adata.obsm['spatial'], distType='euclidean', k=10).main()
sc.pp.filter_genes(adata, min_cells=5)
adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
adata_X = sc.pp.scale(adata_X)
adata_X = sc.pp.pca(adata_X, n_comps=200)
stmap = STMAP_model(input_dim = adata_X.shape[1], 
                        Conv_type='GCNConv',
                        linear_encoder_hidden=[100,20],
                        linear_decoder_hidden=[50,70],
                        conv_hidden=[32,16,8],
                        p_drop=0.1,
                        dec_cluster_n=20,)

train(adata_X, graph_dict, stmap, pre_epochs=200, epochs=200).fit()

"""

import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sknetwork.clustering import Louvain
from sklearn.cluster import SpectralClustering, KMeans
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



class train():
    def __init__(self,
                processed_data,
                graph_dict,
                model,
                pre_epochs,
                epochs,
                corrupt = 0.001,
                lr = 5e-4,
                weight_decay = 1e-4,
                domains = None,
                kl_weight = 100,
                mse_weight = 10, 
                bce_kld_weight = 0.1,
                domain_weight = 1,
                use_gpu = True,
                ):
        if use_gpu:
            self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            print('Using ', self.device)
            

        else:
            self.device = "cpu"
            
        #self.adj_label = graph_dict['adj_label'].to(self.device)
        self.processed_data = processed_data
        self.graph_dict = graph_dict
        #self.data = torch.FloatTensor(processed_data.copy()).to(self.device)   
        #self.adj = graph_dict['adj_norm'].to(self.device)
        
        self.norm = graph_dict['norm_value']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr = lr, weight_decay = weight_decay)
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.num_spots = 256
        self.dec_tol = 0
        self.kl_weight = kl_weight
        self.q_stride = 20
        self.mse_weight = mse_weight
        self.bce_kld_weight = bce_kld_weight
        self.domain_weight = domain_weight
        self.corrupt = corrupt
        if domains is not None:
            self.domains = torch.from_numpy(domains).to(self.device)
        else:
            self.domains = domains
    
    
    
    def pretrain(self, grad_down=5):
      #Create DataLoader for processed matrix
      dataloader = DataLoader(self.processed_data, batch_size=256, shuffle=False)
      #Create DataLoader for graph
      # Step 1: Determine the Batch Size
      batch_size = 256  # Set the batch size as needed based on GPU memory and graph size
      
      # Step 2: Split the Graph Nodes into Batches
      # Example: Split the nodes into equally-sized batches
      num_nodes = self.graph_dict['adj_norm'].size(0)
      num_batches = (num_nodes + batch_size - 1) // batch_size
      node_batches = torch.chunk(torch.arange(num_nodes), chunks=num_batches)
      
      # Step 3: Create Batches of Graph Data
      graph_batches = []
      for nodes in node_batches:
          # Extract the subgraph data for each batch of nodes
          subgraph_adjacency = self.graph_dict['adj_norm'][nodes][:, nodes]
          subgraph_adj_label = self.graph_dict['adj_label'][nodes][:, nodes]
          subgraph_norm_value = self.graph_dict['norm_value']
      
          # Append the subgraph data to the list of graph batches
          graph_batches.append({
              'adj_norm': subgraph_adjacency,
              'adj_label': subgraph_adj_label,
              'norm_value': subgraph_norm_value,
          })
      
      # Step 4: Create a Custom Dataset
      class GraphDataset(Dataset):
          def __init__(self, graph_batches):
              self.graph_batches = graph_batches
      
          def __len__(self):
              return len(self.graph_batches)
      
          def __getitem__(self, idx):
              return self.graph_batches[idx]
  
      # Step 5: Create DataLoader
      graph_dataset = GraphDataset(graph_batches)
      shuffle = False  # Set to False if you want to keep the order of the data
      
      dataloaderG = DataLoader(graph_dataset, batch_size=None, shuffle=shuffle)
      
      with tqdm(total=int(self.pre_epochs), desc="Training an initial model",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
          for epoch in range(self.pre_epochs):
              # Training loop with DataLoader
              for (batch_X,batch_G) in zip(dataloader,dataloaderG):
                  adj_label = batch_G['adj_label'].to(self.device)
                  batch_X = batch_X.float()
                  data = torch.FloatTensor(batch_X.clone()).to(self.device)   
                  adj = batch_G['adj_norm'].to(self.device)
                  inputs_corr = masking_noise(data, self.corrupt)
                  inputs_coor = inputs_corr.to(self.device)
                  self.model.train()
                  self.optimizer.zero_grad()
                  if self.domains is None:
                      z, mu, logvar, de_feat, _, feat_x, gnn_z = self.model(Variable(inputs_coor), adj)
                      preds = self.model.dc(z)
                  else:
                      z, mu, logvar, de_feat, _, feat_x, gnn_z, domain_pred = self.model(Variable(inputs_coor),
                                                                                          adj)
                      preds = self.model.model.dc(z)
                  loss = self.model.deepst_loss(
                      decoded=de_feat,
                      x=data,
                      preds=preds,
                      labels=adj_label,
                      mu=mu,
                      logvar=logvar,
                      n_nodes=256,
                      norm=self.norm,
                      mask=adj_label,
                      mse_weight=self.mse_weight,
                      bce_kld_weight=self.bce_kld_weight,
                  )
                  if self.domains is not None:
                      loss_function = nn.CrossEntropyLoss()
                      domain_loss = loss_function(domain_pred, self.domains)
                      loss += domain_loss * self.domain_weight
                  else:
                      loss = loss
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_down)
                  self.optimizer.step()
                  
                  adj_label.detach()
                  data.detach()
                  adj.detach()
                  inputs_corr.detach()
                  inputs_coor.detach()
              pbar.update(1)
          print('Done pretraining')

    @torch.no_grad()
    def process(self):
      #Create DataLoader for processed matrix
      dataloader = DataLoader(self.processed_data, batch_size=256, shuffle=False)
      #Create DataLoader for graph
      # Step 1: Determine the Batch Size
      batch_size = 256  # Set the batch size as needed based on GPU memory and graph size
      
      # Step 2: Split the Graph Nodes into Batches
      # Example: Split the nodes into equally-sized batches
      num_nodes = self.graph_dict['adj_norm'].size(0)
      num_batches = (num_nodes + batch_size - 1) // batch_size
      node_batches = torch.chunk(torch.arange(num_nodes), chunks=num_batches)
      
      # Step 3: Create Batches of Graph Data
      graph_batches = []
      for nodes in node_batches:
          # Extract the subgraph data for each batch of nodes
          subgraph_adjacency = self.graph_dict['adj_norm'][nodes][:, nodes]
          subgraph_adj_label = self.graph_dict['adj_label'][nodes][:, nodes]
          subgraph_norm_value = self.graph_dict['norm_value']
      
          # Append the subgraph data to the list of graph batches
          graph_batches.append({
              'adj_norm': subgraph_adjacency,
              'adj_label': subgraph_adj_label,
              'norm_value': subgraph_norm_value,
          })
      
      # Step 4: Create a Custom Dataset
      class GraphDataset(Dataset):
          def __init__(self, graph_batches):
              self.graph_batches = graph_batches
      
          def __len__(self):
              return len(self.graph_batches)
      
          def __getitem__(self, idx):
              return self.graph_batches[idx]
  
      # Step 5: Create DataLoader
      graph_dataset = GraphDataset(graph_batches)
      shuffle = False  # Set to False if you want to keep the order of the data
      
      dataloaderG = DataLoader(graph_dataset, batch_size=None, shuffle=shuffle)
      
      self.model.eval()
  
      z_list = []  # List to store the output z tensors for each batch
      q_list = []  # List to store the output q tensors for each batch
  
      with torch.no_grad():
          for (batch_X, batch_G) in zip(dataloader, dataloaderG):
              adj_label = batch_G['adj_label'].to(self.device)
              batch_X = batch_X.float()
              data = torch.FloatTensor(batch_X.clone()).to(self.device)
              adj = batch_G['adj_norm'].to(self.device)
  
              if self.domains is None:
                  z, _, _, _, q, _, _ = self.model(data, adj)
              else:
                  z, _, _, _, q, _, _, _ = self.model(data, adj)
  
              z_list.append(z.cpu().detach().numpy())
              q_list.append(q.cpu().detach().numpy())
  
      # Concatenate the z and q tensors from all batches
      z_result = np.concatenate(z_list, axis=0)
      q_result = np.concatenate(q_list, axis=0)
  
      return z_result, q_result


    def save_model(
        self, 
        save_model_file
        ):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(
        self, 
        save_model_file
        ):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def fit(self, 
            cluster_n=20, 
            clusterType = 'KMeans',
            res = 1.0,
            pretrain = True,
            ):

        """
        load pretrain model for IDEC
        For specific methods, please refer to: https://github.com/IoannisStournaras/Deep-Learning-
                                                for-Deconvolution-of-scRNA-seq-Data    
        """
        #batch_pred=[]
        #batch_real=[]
        #predicted=[]
        #real=[]
        #MSE=[]
        #train_accuracy=[]
        if pretrain:
            print('Pretraining...')
            self.pretrain()
            print('Pretraining done')
            pre_z, _ = self.process()
            print('Done!')
        # z, _, _, _, _, _, _ = self.model(self.data, self.adj)
        # pre_z, _ = self.process()
        
        #Create DataLoader for processed matrix
        dataloader = DataLoader(self.processed_data, batch_size=256, shuffle=False)
        #Create DataLoader for graph
        # Step 1: Determine the Batch Size
        batch_size = 256  # Set the batch size as needed based on GPU memory and graph size
        
        # Step 2: Split the Graph Nodes into Batches
        # Example: Split the nodes into equally-sized batches
        num_nodes = self.graph_dict['adj_norm'].size(0)
        num_batches = (num_nodes + batch_size - 1) // batch_size
        node_batches = torch.chunk(torch.arange(num_nodes), chunks=num_batches)
        
        # Step 3: Create Batches of Graph Data
        graph_batches = []
        for nodes in node_batches:
            # Extract the subgraph data for each batch of nodes
            subgraph_adjacency = self.graph_dict['adj_norm'][nodes][:, nodes]
            subgraph_adj_label = self.graph_dict['adj_label'][nodes][:, nodes]
            subgraph_norm_value = self.graph_dict['norm_value']
        
            # Append the subgraph data to the list of graph batches
            graph_batches.append({
                'adj_norm': subgraph_adjacency,
                'adj_label': subgraph_adj_label,
                'norm_value': subgraph_norm_value,
            })
        
        # Step 4: Create a Custom Dataset
        class GraphDataset(Dataset):
            def __init__(self, graph_batches):
                self.graph_batches = graph_batches
        
            def __len__(self):
                return len(self.graph_batches)
        
            def __getitem__(self, idx):
                return self.graph_batches[idx]
    
        # Step 5: Create DataLoader
        graph_dataset = GraphDataset(graph_batches)
        shuffle = False  # Set to False if you want to keep the order of the data
        
        dataloaderG = DataLoader(graph_dataset, batch_size=None, shuffle=shuffle)
        
        if clusterType == 'KMeans':
            cluster_method = KMeans(n_clusters= cluster_n, n_init= cluster_n * 2, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
        elif clusterType == 'Louvain':
            cluster_data = sc.AnnData(pre_z)
            sc.pp.neighbors(cluster_data, n_neighbors=cluster_n)
            sc.tl.louvain(cluster_data, resolution = res)
            y_pred_last = cluster_data.obs['louvain'].astype(int).to_numpy()
            n_clusters = len(np.unique(y_pred_last))
            features = pd.DataFrame(pre_z,index=np.arange(0,pre_z.shape[0]))
            Group = pd.Series(y_pred_last,index=np.arange(0,features.shape[0]),name="Group")
            Mergefeature = pd.concat([features,Group],axis=1)
            cluster_centers_ = np.asarray(Mergefeature.groupby("Group").mean())
            if self.domains is None:
                self.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)
            else:
                self.model.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)
        
        
        with tqdm(total=int(self.pre_epochs), desc="Training a final model", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
          for epoch in range(self.epochs):
              #batch_pred=[]
              #batch_real=[]
              if epoch % self.q_stride == 0:
                  _, q = self.process()
                  q = self.model.target_distribution(torch.Tensor(q).clone().detach())
                  y_pred = q.cpu().numpy().argmax(1)
                  delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                  y_pred_last = np.copy(y_pred)
                  self.model.train()
                  if epoch > 0 and delta_label < self.dec_tol:
                      print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                      print('Reached tolerance threshold. Stopping training.')
                      break
              #print(q)
              dataloaderQ = DataLoader(q, batch_size=256, shuffle=False)
              torch.set_grad_enabled(True)
              self.model.train()
  
              for (batch_X, batch_G, batch_q) in zip(dataloader, dataloaderG,dataloaderQ):
                  adj_label = batch_G['adj_label'].to(self.device)
                  batch_X = batch_X.float()
                  data = torch.FloatTensor(batch_X.clone()).to(self.device)
                  adj = batch_G['adj_norm'].to(self.device)
                  inputs_corr = masking_noise(data, self.corrupt)
                  inputs_coor = inputs_corr.to(self.device)
                
                  if self.domains is None:
                      z, mu, logvar, de_feat, out_q, feat_x, gnn_z = self.model(Variable(inputs_coor), adj)
                      preds = self.model.dc(z)
                  else:
                      z, mu, logvar, de_feat, out_q, feat_x, gnn_z, domain_pred = self.model(Variable(inputs_coor), adj)
                      loss_function = nn.CrossEntropyLoss()
                      domain_loss = loss_function(domain_pred, self.domains)
                      preds = self.model.model.dc(z)
                #self.adj.to('cuda:2')
                #import gc
                #gc.collect()
                #torch.cuda.empty_cache()
                #self.adj_label.to('cuda:0')
                  batch_q = batch_q.float()
                  loss_deepst = self.model.deepst_loss(
                                decoded = de_feat, 
                                x = data, 
                                preds = preds, 
                                labels = adj_label, 
                                mu = mu, 
                                logvar = logvar, 
                                n_nodes = 256, 
                                norm = self.norm, 
                                mask = adj_label, 
                                mse_weight = self.mse_weight, 
                                bce_kld_weight = self.bce_kld_weight
                                )
                  
                  loss_kl = F.kl_div(out_q.log(), batch_q.to(self.device))
                  if self.domains is None:
                      loss = self.kl_weight * loss_kl + loss_deepst
                  else:
                      loss = self.kl_weight * loss_kl + loss_deepst + domain_loss
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                  self.optimizer.step()
                  #batch_pred.append(preds)
                  #batch_real.append(data)
                  
  
                  # Detach tensors to free GPU memory
                  adj_label.detach()
                  data.detach()
                  adj.detach()
                  inputs_corr.detach()
                  inputs_coor.detach()
              pbar.update(1)
              #batch_real=torch.cat(batch_real, dim=0).cpu().numpy()
              #batch_pred=torch.cat(batch_pred, dim=0).cpu().numpy()
              #print('batch', batch_pred)
              #mse = np.mean((batch_real - batch_pred)**2)
              #MSE.append(mse)

          print('Done training')
          
          #plt.plot(range(len(MSE)), MSE)
          #plt.xlabel('Epoch')
         # plt.ylabel('MSE')
          #plt.title('MSE error')
         # plt.show()

           
         
def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise