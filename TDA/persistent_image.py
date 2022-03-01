
#%%
import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform
import dionysus as d
import matplotlib.pyplot as plt
import time
from ripser import ripser
from persim import plot_diagrams, PersImage
path = os.getcwd()



# PI generation #
def persistence_images(dgms, resolution = [200,200], return_raw = False, normalization = True, bandwidth = 1., power = 1., dimensional = 0):
  PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in dgms]), np.vstack([dgm[:, 1:2] for dgm in dgms])
  if PXs.shape[0]!=0:
      
      xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
      x = np.linspace(xm, xM, resolution[0])
      y = np.linspace(ym, yM, resolution[1])
      X, Y = np.meshgrid(x, y)
      Zfinal = np.zeros(X.shape)
      X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

      # Compute image
      P0, P1 = np.reshape(dgms[int(dimensional)][:, 0], [1, 1, -1]), np.reshape(dgms[int(dimensional)][:, 1], [1, 1, -1])
      weight = np.abs(P1 - P0)
      distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

      if return_raw:
          lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
          lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
      else:
          weight = weight ** power
          Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

      output = [lw, lsum] if return_raw else Zfinal

      if normalization:
          norm_output = (output - np.min(output))/(np.max(output) - np.min(output))
      else:
          norm_output = output
  else:
       norm_output = np.zeros(resolution)

  return norm_output

# PI generation from library #
# https://github.com/scikit-tda/persim/blob/master/docs/notebooks/Persistence%20images.ipynb

# save ATD PIs as .npy

folderSaveImgs = 'data/COVID_TX_PERCENT/PI_TEST/'
nameFolderNet = 'data/COVID_TX_PERCENT/output_test/DATA/BCALL' # * Example 1
TotalNets =175 # Total of graphs in the dynamic network
PIs = []
for kNet in range(0,TotalNets): 
  csv = np.genfromtxt (nameFolderNet+str(kNet)+".txt")
  csv = [csv]
  #csv = csv[csv[:,0]==0,]
  #csv = [csv[csv[:,2]>0,1:3]]
  PI = persistence_images(dgms= csv)
  PIs.append(PI)

np.save(folderSaveImgs+'ATD_PIs', np.array(PIs))
