# Created by Karla Paniagua
# August 2025
# Spatial Object

from anndata import AnnData
import scanpy as sc
import squidpy as sq
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os 
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def spatial_object(df,exp,img,name):
    # df is the metadata file
    # exp is the expression file
    # img is the path to the stitched image
    # name of the tissue
    exp = exp.drop(columns=[col for col in exp.columns if "NegPrb" in col])
    exp = exp.drop(columns=[col for col in exp.columns if "SystemControl" in col])
    exp = exp.drop(columns=[col for col in exp.columns if "Negative" in col])

    counts = exp
    coordinates=df[['CenterX_global_px', 'CenterY_global_px']]
    coordinates=np.array(coordinates[['CenterX_global_px', 'CenterY_global_px']])

    image=mpimg.imread(img)
    adata = AnnData(counts, obsm={"spatial": coordinates})

    spatial_key = "spatial"
    library_id = name
    adata.uns[spatial_key] = {library_id: {}}
    adata.uns[spatial_key][library_id]["images"] = {}
    adata.uns[spatial_key][library_id]["images"] = {"hires": image}
    adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 3}

    coordinates=df[['CenterX_global_px', 'CenterY_global_px']]
    adata.obs['CenterPatchX']=coordinates['CenterX_global_px'].values
    adata.obs['CenterPatchY']=coordinates['CenterY_global_px'].values
    adata.obs['imagerow']=adata.obs['CenterPatchX']
    adata.obs['imagecol']=adata.obs['CenterPatchY']
    adata.obs['array_row']=adata.obs.CenterPatchX
    adata.obs['array_col']=adata.obs.CenterPatchY
    return adata

