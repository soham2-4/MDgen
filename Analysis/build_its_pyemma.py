#!/home/jmondal/msahil/0_softwares/anaconda3/bin/python3

################################################################################################################################################################################################################################################

# OWNER : SUBINOY_ADHIKARI 

################################################################################################################################################################################################################################################

import os, gc, sys, stat
import pyemma
import numpy as np
import pickle as pkl
import pyemma.msm as msm
import pyemma.plots as mplt
import matplotlib.pyplot as plt
from colorama import Fore, Style
import pyemma.coordinates as coor
from sklearn.decomposition import PCA
from pyemma.util.contexts import settings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

################################################################################################################################################################################################################################################

# Method to scale the data
def build_scaled_data(data, scaler=None):
    return scaler().fit_transform(data)

# Method to reshape the data
def build_reshape_data(data, traj_shape):
    data_reshaped = []
    start_frame=0
    for i in range(len(traj_shape)):    
        end_frame=start_frame+traj_shape[i]
        data_reshaped.append(data[int(start_frame):int(end_frame)])     
        # Update start frame #
        start_frame=start_frame+traj_shape[i] 
    return data_reshaped

# Method for k-means clustering
def build_clustering(data=None,
            n_clusters=100, 
            max_iter=10000,
            tolerance=1e-4, 
            stride=1,
            metric='euclidean', 
            init_strategy='uniform',
            fixed_seed=73,  
            n_jobs=32,
            chunk_size=5000,
            skip=0,                    
            filename_init="test"):

    with settings(show_progress_bars=True):
        clu=coor.cluster_kmeans(data,
                    k=n_clusters,
                    max_iter=max_iter, 
                    fixed_seed=fixed_seed, 
                    metric=metric,
                    tolerance=tolerance,
                    init_strategy=init_strategy, 
                    stride=stride)     
    # Saving the clustering 
    pkl.dump(clu, open(f"{filename_init}_nclu_{n_clusters}_kmeans.pkl", "wb"))    
    print(f"{Fore.GREEN} CREATED FILE FOR KMEANS CLUSTERING :{Fore.MAGENTA} {filename_init}_nclu_{n_clusters}_kmeans.pkl {Style.RESET_ALL} \n")
    return clu

# Method to build the dtrajs
def build_dtrajs(clu, n_clusters=None, filename_init="test"):
    dtrajs = clu.dtrajs
    pkl.dump(dtrajs, open(f"{filename_init}_nclu_{n_clusters}_dtrajs.pkl", "wb"))
    print(f"{Fore.GREEN} CREATED FILE FOR DISCRETE TRAJECTORY :{Fore.MAGENTA} {filename_init}_nclu_{n_clusters}_dtrajs.pkl {Style.RESET_ALL} \n")
    return dtrajs

# Method to determine the implied timescales        
def build_implied_timescale(dtrajs=None, lags=None, errors=None):
    its=msm.its(dtrajs=dtrajs, lags=lags, errors=errors)
    return its


# Method for plotting the implied timescales
def build_figure(its=None, nits=None, ylog=True, n_clusters=None, filename_init="test", timestep=0.01, units='ns'):
    fig, ax = plt.subplots(sharex=False, sharey=True, 
                       gridspec_kw={'hspace':0, 'wspace':0.02}, figsize=(10,6))
    plt.rcParams['axes.linewidth'] = 2

    mplt.plot_implied_timescales(its, ax=ax, nits=nits, lw=2, ylog=ylog, dt=timestep, units=units)

    # FIX: set Y-axis range to show up to 10^4 ns
    ax.set_ylim(1, 1e4)
    # Optional dynamic alternative (uncomment to use):
    # max_ts = np.max(its.timescales) if hasattr(its, 'timescales') else 1e4
    # ax.set_ylim(1, max(1e4, max_ts * 1.2))

    ax.tick_params(axis='both', which='major', top=True, right=True, direction='in', length=10, width=2, pad=10, labelsize=20)
    ax.tick_params(axis='y', which='minor', top=True, right=True, direction='in', length=5, width=2, pad=10, labelsize=0)
    ax.locator_params(axis='x', nbins=10)
    ax.set_xlabel("lagtime (ns)", size=20)     
    ax.set_ylabel("timescale (ns)", size=20)        
    ax.set_title(f"Microstates = {n_clusters}", y=1.0, fontsize=20)
    ax.ticklabel_format(axis='x', style='sci', scilimits=None, useOffset=None, useLocale=True, useMathText=None)
    # ax.legend(loc=1, fontsize=25,)
    # ax.set_xlim(1, 3000)
    # # ax.set_ylim(0,10000)
    # ax.grid(False)#, ls='dashed', c='gray')
    # ax.axvline(50, lw=3, c='m')
    plt.savefig(f"{filename_init}_nclu_{n_clusters}_its.svg", dpi=300, format='svg', bbox_inches='tight')
    print(f"{Fore.GREEN} CREATED FILE FOR ITS PLOT :{Fore.MAGENTA} {filename_init}_nclu_{n_clusters}_its.svg {Style.RESET_ALL} \n")
    plt.savefig(f"{filename_init}_nclu_{n_clusters}_its.png", dpi=300, format='png', bbox_inches='tight')    
    print(f"{Fore.GREEN} CREATED FILE FOR ITS PLOT :{Fore.MAGENTA} {filename_init}_nclu_{n_clusters}_its.png {Style.RESET_ALL} \n")    
    return None
    
################################################################################################################################################################################################################################################

if __name__ == "__main__":

    # Protein/System name
    filename_init="toy_gen"
    FILENAME_INIT="toy_gen"

    #stride=10

    timestep=1 # timestep in ns #
    
    # Filename containing the reduced dimension data to build the Markov Model
    reduced_dim_file="toy.npy"

    # Filename containing the length of trajectory in each of the replicas
    #traj_length_file=pkl.load(open(f"../../../{filename_init}_trajectory_length_info.pkl", 'rb'))
    #traj_length_file=np.load("../shape_generated_all.npy")

    # Load the reduced dimension data to build the Markov Model
    data = np.load(reduced_dim_file)
    #data = PCA().fit_transform(data)
    #data=np.column_stack([data[:,0], data[:,1]])
    #data=StandardScaler().fit_transform(data)

    # Scale the data (if required)
    #data=build_scaled_data(data, None)

    # Reshape the data
    #traj_shape = np.array([traj_length_file[key] for key in traj_length_file.keys()])
    #traj_shape = np.asarray(traj_length_file)
    #print(np.sum(traj_shape))    
    #data = build_reshape_data(data, traj_shape)
    #print(len(np.concatenate(data)))
    print(np.shape(data))

    # Create a loader by specifying the source file 
    data = coor.source(data)

    # Clustering the data
    #n_clusters=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000]
    #n_clusters=[50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    #n_clusters=[50, 60, 70, 80, 100, 120, 140, 160, 180, 200]
    n_clusters=[50, 100, 200, 300, 400, 500]

    # Lag times for which the implied timescales need to be determined
    #lags = [1,2,5,10,15,20,25,30,40,50,60,80,100,200,300,500,600,800,1000,1200,1400,1600,1800,2000]
    #lags = [1,2,5,10,15,20,25,30,40,50,60,80,100,200,300,500,1000,2000,5000,10000,20000,25000,30000]
    #lags = [1,2,5,10,15,20,25,50,75,100,125,150,175,200]
    lags = [1,2,5,10,15,20,25,30,40,50,60,80,100,200,300,500]
    
    
    # Build the markov model for different number of clusters
    for i in range(len(n_clusters)):
        clu = build_clustering(data=data,
                    n_clusters=n_clusters[i], 
                    max_iter=100000,
                    tolerance=1e-5, 
                    stride=1,
                    metric='euclidean', 
                    init_strategy='uniform',
                    fixed_seed=73,  
                    n_jobs=128,
                    chunk_size=100,
                    skip=0,                    
                    filename_init=filename_init)    
                    
        # Build the discrete trajectories                
        dtrajs = build_dtrajs(clu, n_clusters=n_clusters[i], filename_init=filename_init)
        
        # Build the implied timescales plot
        its=build_implied_timescale(dtrajs=dtrajs, lags=lags, errors=None)
                    
        # Saving the ITS plot
        build_figure(its=its, 
                 nits=10, 
                 ylog=True, 
                 n_clusters=n_clusters[i], 
                 filename_init=filename_init,
                 timestep=timestep)


    # Free the memory and quit
    for element in dir():
        if element[0:2] != "__":
            del element
    gc.collect()
    sys.exit(0)

################################################################################################################################################################################################################################################
