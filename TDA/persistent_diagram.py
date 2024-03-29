
#%% Libraries and Functions
import dionysus as d
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import os
import pandas as pd
import time



from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')

"""# New Section

# New Section
"""

#%% Parameters 
# TDA 
scaleParameter = 1.0 # Scale Parameter
maxDimHoles = 3 # Maximum Dimension of Holes (always choose the desired number + 1)
# Dynamic Network
nameFolderNet = 'data/COVID_TX_PERCENT/ZIGZAG_DATA_TRAIN/filename' # * filelocation

NVertices = 127 # Number of vertices, all graphs have the same number of vertices
TotalNets = 3489 # Total of graphs in the dynamic network
folderSaveImgs = 'data/COVID_TX_PERCENT/output_test/' # HERE !!! Output...

#%% Creating folders and compute everything automatically
start_time = time.time() # **** To measure time
for kNet in range(0,TotalNets): 
    #kNet = 0 # Network's index
    print('*************  '+str(kNet)+'  *************')
    #%% Open all sets (point-cloud/Graphs)
    print("Loading data...") # Beginning
    Graphs = []
    #edgesList = np.loadtxt(nameFolderNet+str(kNet+1)+".txt") # Load data (TXT)
    edgesList = np.loadtxt(nameFolderNet+str(kNet)+".csv", delimiter=',') # Load data (CSV)
    Graphs.append(edgesList)
    print("  --- End Loading...") # Ending

    #%% Plot Graph
    print("Plot and NetworkX...") # Beginning
    GraphsNetX = []
    plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    g = nx.Graph()
    g.add_nodes_from(list(range(1,NVertices+1))) # Add vertices...
    i = 0 # There is only one graph... the current one...
    if(Graphs[i].ndim==1 and len(Graphs[i])>0):
        g.add_edge(Graphs[i][0], Graphs[i][1], weight=Graphs[i][2])
    elif(Graphs[i].ndim==2):
        for k in range(0,Graphs[i].shape[0]):
            g.add_edge(Graphs[i][k,0], Graphs[i][k,1], weight=Graphs[i][k,2])
    GraphsNetX.append(g)
    plt.title(str(kNet))
    pos = nx.circular_layout(GraphsNetX[i])
    nx.draw(GraphsNetX[i], pos, node_size=15, edge_color='r') 
    #nx.draw_circular(GraphsNetX[i], node_size=15, edge_color='r') 
    labels = nx.get_edge_attributes(GraphsNetX[i], 'weight')
    for lab in labels:
        labels[lab] = round(labels[lab],4)
    nx.draw_networkx_edge_labels(GraphsNetX[i], pos, edge_labels=labels,label_pos=0.2,font_size=10)

    plt.savefig(folderSaveImgs+'GRAPHS/Graphs'+str(kNet)+'.pdf', bbox_inches='tight')
    plt.close()
    print("---- End: Plot and NetworkX...") # End

    #%% Building unions and computing distance matrices 
    print("Computing distance matrices...") # Beginning 
    MDisGraph = [] 
    i = 0 # There is only one graph... the current one...
    # --- To build the distance matrix (from 0 to 1.0)
    MDisAux = np.zeros((NVertices, NVertices))
    A = nx.adjacency_matrix(GraphsNetX[i]).todense()
     #### This number is very important !!! *******
    # It should be bigger that: scaleParameter
    A[A==0] = 1.1  
    A[range(NVertices), range(NVertices)] = 0 # To set the diagonal to 0 
    MDisAux[0:NVertices,0:NVertices] = A 
    # --- To build the distance matrix (from 1.0 to 0 - inverted) 
    # MDisAux = np.zeros((NVertices, NVertices))
    # A = nx.adjacency_matrix(GraphsNetX[i]).todense()
    # indGreater0 = np.where(A>0) 
    # A[indGreater0] = 1.0 - A[indGreater0]
    # A[A==0] = 1.1  #### This number is very important !!! *******
    # A[range(NVertices), range(NVertices)] = 0 # To set the diagonal to 0 
    # MDisAux[0:NVertices,0:NVertices] = A 
    # --- Distance in condensed form 
    pDisAux = squareform(MDisAux) 
    # --- To save distances 
    MDisGraph.append(pDisAux) # To save distance matrix
    print("  --- Distance matrices...") # Ending


    #%% To perform Ripser computations
    print("Computing Vietoris-Rips complexes...") # Beginning 
    GVRips = [] 
    i = 0 # There is only one graph... the current one...
    ripsAux = d.fill_rips(MDisGraph[i], maxDimHoles, scaleParameter) 
    GVRips.append(ripsAux)
    print("  --- End Vietoris-Rips computation") # Ending 

    #%% To perform Ripser computations
    print("Persistence Homology...") # Beginning 
    GPHomology = [] 
    Gdgms = [] 
    i = 0 # There is only one graph... the current one...
    phAux = d.homology_persistence(GVRips[i])
    GPHomology.append(phAux) 
    dgmsAux = d.init_diagrams(GPHomology[i], GVRips[i])
    Gdgms.append(dgmsAux)
    print("  --- End Persistence Homology") # Ending 

    #%% Diagram 
    #-- Important: It does not show information of intervals (p.birth, Inf)
    # i = 0 # There is only one graph... the current one...
    # d.plot.plot_diagram(Gdgms[i][0], show = True)
    # d.plot.plot_bars(Gdgms[i][0], show = True)

    # %% Personalized plot
    theta = 0 # There is only one graph... the current one...
    for i,dgm in enumerate(Gdgms[theta]):
        print("Dimension:", i) 
        plt.figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
        if(i<maxDimHoles): 
            BCfull = np.zeros((len(dgm), 3)) 
            matBarcode = np.zeros((len(dgm), 2)) 
            k = 0
            for p in dgm:
                BCfull[k,0] = i
                BCfull[k,1] = p.birth
                BCfull[k,2] = p.death
                #print("( "+str(p.birth)+"  "+str(p.death)+" )") 
                matBarcode[k,0] = p.birth
                matBarcode[k,1] = p.death
                k = k + 1
            BCfull[BCfull==np.inf] = -1 # Change infty to -1
            matBarcode[matBarcode==np.inf] = 1.0 # Change infty to 1.0 
            #matBarcode = matBarcode
            #print(matBarcode)
            for j in range(0,matBarcode.shape[0]): 
                plt.plot(matBarcode[j], [j,j], 'b') 
            #Human readable data
            if(i==0):
                BCALL = BCfull
            else:
                BCALL = np.concatenate((BCALL, BCfull),axis=0)
            np.savetxt(folderSaveImgs+'DATA/BCALL'+str(kNet)+'.txt', BCALL)

            #plt.xticks(np.arange(sizeWindow))
            #plt.grid(axis='x', linestyle='-')
            plt.savefig(folderSaveImgs+'BARCODES/WDW'+str(kNet)+'BoxPlot'+str(i)+'.pdf', bbox_inches='tight') 
            plt.close()
            #plt.show()

# *** Timing
print("\nTIME: "+str((time.time() - start_time))+" Seg ---  "+str((time.time() - start_time)/60)+" Min ---  "+str((time.time() - start_time)/(60*60))+" Hr ")
 
#%%

