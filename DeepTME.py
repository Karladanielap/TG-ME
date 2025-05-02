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
from trainer import train

from augment import augment_adata
import gc
import time
import torch.nn as nn
import torch.distributed as dist


class run():
  def __init__(
    self,
    verbose=True,
    use_gpu = True,
    ):


    self.verbose = verbose
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
    adata_augment = augment_adata(ct,adata, adjacent_weight = adjacent_weight, neighbour_k = neighbour_k, platform = "Visium", weights = weights, spatial_k = spatial_k,)
    print("Step 1: Augment gene representation is Done!")
    return adata_augment
   
