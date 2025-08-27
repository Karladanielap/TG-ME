# Created by Karla Paniagua
# August 2025
# Stitching coordinates functions 

from stitch2d import StructuredMosaic
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import warnings
warnings.filterwarnings('ignore')

from PIL import Image 

 #For modifying the coordinates to plot in the stitched image input the dataframe with the coordinates 
#and the number of images per row -1
def modify_coordinates_raster(df,imrow,x,y,firstFOV):
    #df is the metadata of your file.
    #imrow is the number of images per row -1
    #x is the width of the image.
    #y is the height og the image.
    #firstFOV is the name of the first FOV in the tissue.
    v1=0
    v2=0
    for j in df['fov'].unique():
        print('Calculating for image: ',j)
        a=df[df['fov']==j]
        if j==firstFOV:
            df.loc[df['fov']==j,'CenterX_global_px']=df.loc[df['fov']==j,'CenterX_local_px']
            df.loc[df['fov']==j,'CenterY_global_px']=df.loc[df['fov']==j,'CenterY_local_px']
            v1=v1+1
        elif v1>0 & v2>=0:       
            df.loc[df['fov']==j,'CenterX_global_px']=df.loc[df['fov']==j,'CenterX_local_px']+(v1*x)
            df.loc[df['fov']==j,'CenterY_global_px']=df.loc[df['fov']==j,'CenterY_local_px']+(v2*y)
            if v1==imrow:
                v1=0
                v2=v2+1
            else:
                v1=v1+1
        elif v1==0 & v2>1:
            df.loc[df['fov']==j,'CenterX_global_px']=df.loc[df['fov']==j,'CenterX_local_px']
            df.loc[df['fov']==j,'CenterY_global_px']=df.loc[df['fov']==j,'CenterY_local_px']+(v2*y)
        elif v1>=0 & v2>=0:
            df.loc[df['fov']==j,'CenterX_global_px']=df.loc[df['fov']==j,'CenterX_local_px']+(v1*x)
            df.loc[df['fov']==j,'CenterY_global_px']=df.loc[df['fov']==j,'CenterY_local_px']+(v2*y)
            if v1==imrow:
                v1=0
                v2=v2+1
            else:
                v1=v1+1
    print('Done!')
    return(df)


#For modifying the coordinates to plot in the stitched image input the dataframe with the coordinates 
#and the number of images per row -1
def modify_coordinates_snake(df,imrow,x,y,firstFOV):
    v1=0
    v2=0
    for j in df['fov'].unique():
        print('Calculating for image: ',j)
        a=df[df['fov']==j]
        if j==firstFOV:
            df.loc[df['fov']==j,'CenterX_global_px']=df.loc[df['fov']==j,'CenterX_local_px']  
            df.loc[df['fov']==j,'CenterY_global_px']=df.loc[df['fov']==j,'CenterY_local_px']
            v1=v1+1

        elif (v1>0) & (v2<1):
            df.loc[df['fov']==j,'CenterX_global_px']=df.loc[df['fov']==j,'CenterX_local_px']
            df.loc[df['fov']==j,'CenterY_global_px']=df.loc[df['fov']==j,'CenterY_local_px']+(v1*y)
            if v1==imrow:
                v2=v2+1
            else:
                v1=v1+1
        elif (v1>=0) & (v2==1):     
            df.loc[df['fov']==j,'CenterX_global_px']=df.loc[df['fov']==j,'CenterX_local_px']+(v2*y)        
            df.loc[df['fov']==j,'CenterY_global_px']=df.loc[df['fov']==j,'CenterY_local_px']+(v1*y)
            if v1==0:
                v2=v2+1
            if v1>0:
                v1=v1-1          
        elif (v1>=0) & (v2==2):       
            df.loc[df['fov']==j,'CenterX_global_px']=df.loc[df['fov']==j,'CenterX_local_px']+(v2*y)         
            df.loc[df['fov']==j,'CenterY_global_px']=df.loc[df['fov']==j,'CenterY_local_px']+(v1*y)
            v1=v1+1
            
    print('Done!')
    return(df)

