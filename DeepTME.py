import os
import psutil
import time
import torch
import math
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata
from pathlib import Path
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import distance

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Union, Callable

from utils_func import *
from his_feat import image_feature, image_crop
from adj import graph, combine_graph_dict
from model import DeepST_model, AdversarialNetwork
from trainer import train

from augment import augment_adata
import gc
import time
import torch.nn as nn
import torch.distributed as dist


class run():
  def __init__(
    self,
    save_path="./",
    pre_epochs=1000, 
    epochs=500,
    pca_n_comps = 200,
    linear_encoder_hidden=[32,20],
    linear_decoder_hidden=[32],
    conv_hidden=[32,8],
    verbose=True,
    platform='Visium',
    cnnType='ResNet50',
    Conv_type='GCNConv',		
    p_drop=0.01,
    dec_cluster_n=20,
    n_neighbors=15,
    min_cells=3,
    grad_down = 5,
    kl_weight = 100,
    mse_weight = 10, 
    bce_kld_weight = 0.1,
    domain_weight = 1,
    use_gpu = True,
    ):
    self.save_path = save_path
    self.pre_epochs = pre_epochs
    self.epochs = epochs
    self.pca_n_comps = pca_n_comps
    self.linear_encoder_hidden = linear_encoder_hidden
    self.linear_decoder_hidden = linear_decoder_hidden
    self.conv_hidden = conv_hidden
    self.verbose = verbose
    self.platform = platform
    self.cnnType = cnnType
    self.Conv_type = Conv_type
    self.p_drop = p_drop
    self.dec_cluster_n = dec_cluster_n
    self.n_neighbors = n_neighbors
    self.min_cells = min_cells
    self.platform = platform
    self.grad_down = grad_down
    self.kl_weight = kl_weight
    self.mse_weight = mse_weight
    self.bce_kld_weight = bce_kld_weight
    self.domain_weight = domain_weight
    self.use_gpu = use_gpu
  def _get_graph(
    self,data,distType = "Radius",k = 12,rad_cutoff = 150,):
    graph_dict = graph(data, distType=distType, k=k, rad_cutoff=rad_cutoff).main()
    print("Step 2: Graph computing is Done!")
    return graph_dict
  
  def _get_augment(
    self,
    ct,adata,
    adjacent_weight = 0.3,
    neighbour_k = 4,
    weights = "weights_matrix_all",
    spatial_k = 30,
    ):
    adata_augment = augment_adata(ct,adata, adjacent_weight = adjacent_weight, neighbour_k = neighbour_k, platform = self.platform, weights = weights, spatial_k = spatial_k,)
    print("Step 1: Augment gene representation is Done!")
    return adata_augment
   
  def _optimize_cluster(
    self,
    adata,
    resolution: list = list(np.arange(1, 1.5, 0.1)),
    ):
    scores = []
    for r in resolution:
      sc.tl.leiden(adata, resolution=r)
      s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
      scores.append(s)
    cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
    best_idx = np.argmax(cl_opt_df["score"])
    res = cl_opt_df.iloc[best_idx, 0]
    print("Best resolution: ", res)
    return res
     
  def _priori_cluster(
    self,
    adata,
    n_domains =7,
    ):
    for res in sorted(list(np.arange(0.1, 0.2, 0.01)), reverse=True):
      sc.tl.leiden(adata, random_state=0, resolution=res)
      count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
      if count_unique_leiden == n_domains:
        break
    print("Best resolution: ", res)
    return res
   
   
  def _fit(
    self,
    adata, 
    graph_dict,
    domains = None,
    dim_reduction = False,
    pretrain = True,
    save_data = False,
    ):
    print("Your task is processing, please wait")
    start_time = time.time()

    #################### data preprocess
    if self.platform in ["Visium", "ST", "slideseqv2", "stereoseq"]:
      adata.X = adata.obsm["augment_gene_data"].astype(float)
      print('Agumented')
      if dim_reduction:
        #sc.pp.filter_genes(adata, min_cells = self.min_cells)
        #print('Filtered')
        adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
        print('Normalized')
        adata_X = sc.pp.log1p(adata_X)
        print('log')
        adata_X = sc.pp.scale(adata_X)
        print('scaled')
        concat_X = sc.pp.pca(adata_X, n_comps=self.pca_n_comps)
        print('PCA')
      else:
        #sc.pp.filter_genes(adata, min_cells = self.min_cells)
        #print('Filtered genes')
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        print('HVG done')
        sc.pp.normalize_total(adata, exclude_highly_expressed=True, inplace=False)
        print('Normalized')
        sc.pp.log1p(adata)
        sc.pp.scale(adata)
        print('Finished log and scale')
        #sc.tl.diffmap(adata)
        #concat_X = adata.obsm['X_diffmap'] #diffmaps
        concat_X = adata[:, adata.var['highly_variable']].X
        #from sklearn.decomposition import FastICA
        #ica = FastICA(n_components=300,whiten="arbitrary-variance")
        #X=adata.X
        #concat_X = ica.fit_transform(X)
    else:
      concat_X = adata.obsm["augment_gene_data"]
    #################### load model
    print(concat_X.shape)
    deepst_model = DeepST_model(
       input_dim = concat_X.shape[1], Conv_type = self.Conv_type,linear_encoder_hidden= self.linear_encoder_hidden,linear_decoder_hidden= self.linear_decoder_hidden,conv_hidden= self.conv_hidden,p_drop=self.p_drop,dec_cluster_n=self.dec_cluster_n,)
    print('Done model definition')

    
    if domains is None:	
      deepst_training = train(concat_X, graph_dict, deepst_model, pre_epochs=self.pre_epochs, epochs=self.epochs,kl_weight = self.kl_weight,mse_weight = self.mse_weight, bce_kld_weight = self.bce_kld_weight,domain_weight = self.domain_weight,use_gpu = self.use_gpu,)
      print('training set')
     

    else:
      deepst_adversial_model = AdversarialNetwork(model = deepst_model, n_domains = int(len(self.data_name)))
      deepst_training = train(concat_X, graph_dict, deepst_adversial_model, pre_epochs=self.pre_epochs, epochs=self.epochs, kl_weight = self.kl_weight,mse_weight = self.mse_weight, bce_kld_weight = self.bce_kld_weight,domain_weight = self.domain_weight,domains = domains,use_gpu = self.use_gpu,)
    if pretrain:
      print('Start pretaining')
      deepst_training.fit()
    else:
      deepst_training.pretrain(grad_down = self.grad_down)
    deepst_embed, _ = deepst_training.process()
    if self.verbose:
      print("Step 3: DeepST training has been Done!")
    ##### save deep learning embedding 
    adata.obsm["DeepST_embed"] = deepst_embed
    print(u'Current memory usage：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time / 60 :.2f} minutes")
    print("Your task has been completed, thank you")
    print("Of course, you can also perform downstream analysis on the processed data")
    return adata
   
  def _get_cluster_data(
    self,
    adata,
    n_domains,
    priori = True,
    ):
    print('Getting microenvironment clusters')
    print('Calculating neighbors...')
    sc.pp.neighbors(adata, use_rep='DeepST_embed', n_neighbors = self.n_neighbors)
    
    if priori:
      res = self._priori_cluster(adata, n_domains=n_domains)
    else:
      res = self._optimize_cluster(adata)
    print('Calculating leiden algorithm...')
    sc.tl.leiden(adata, key_added="DeepST_domain", resolution=res)
    ######### Strengthen the distribution of points in the model
    print('Calculating adjacency matrix')
    adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
    refined_pred= refine(sample_id=adata.obs.index.tolist(), 
    pred=adata.obs["DeepST_domain"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["DeepST_refine_domain"]= refined_pred
    # save_data_path = Path(os.path.join(self.save_path, 'Data', f'{self.data_name}'))
    # save_data_path.mkdir(parents=True, exist_ok=True)
    # adata.write(os.path.join(save_data_path, f'{self.data_name}_processed.h5ad'), compression="gzip")
    print('Done')
    return adata
    
    
  def plot_domains(self, 
    adata, 
    data_name,
    img_key=None, 
    color='DeepST_refine_domain',
    show=False,
    legend_loc='right margin',
    legend_fontsize='x-large',
    size=1.6,
    dpi=300):
    if isinstance(data_name, str):
      sc.pl.spatial(adata, img_key=img_key, color=color, show=show, 
        legend_loc=legend_loc, legend_fontsize=legend_fontsize, spot_size=size)
      save_path_figure = Path(os.path.join(self.save_path, "Figure", data_name))
      save_path_figure.mkdir(parents=True, exist_ok=True)
      plt.savefig(os.path.join(save_path_figure,f'{data_name}_domains.pdf'), bbox_inches='tight', dpi=dpi)
    else:
      pass
       
  def plot_umap(self, 
    adata,
    data_name,
    color='DeepST_refine_domain', 
    legend_loc=None,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    add_outline=True,
    dpi=300,
    ):
    umap_adata = anndata.AnnData(adata.obsm["DeepST_embed"])
    umap_adata.obs_names = adata.obs_names
    umap_adata.obs = adata.obs
    sc.pp.neighbors(umap_adata, n_neighbors = self.n_neighbors)
    sc.tl.umap(umap_adata)
    sc.pl.umap(umap_adata, color=color, add_outline=add_outline, legend_loc=legend_loc,
    legend_fontsize=legend_fontsize, legend_fontoutline=legend_fontoutline, 
    frameon=frameon)
    if isinstance(data_name, list):		
      save_path_figure = Path(os.path.join(self.save_path, "Figure", '_'.join(data_name)))
      save_path_figure.mkdir(parents=True, exist_ok=True)
      plt.savefig(os.path.join(save_path_figure, f"{'_'.join(data_name)}_umap.pdf"), bbox_inches='tight', dpi=dpi)
    else:
      save_path_figure = Path(os.path.join(self.save_path, "Figure", data_name))
      save_path_figure.mkdir(parents=True, exist_ok=True)
      plt.savefig(os.path.join(save_path_figure, f"{data_name}_umap.pdf"), bbox_inches='tight', dpi=dpi)
