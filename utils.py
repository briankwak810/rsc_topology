import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from trajectory import rat_trajectory_PV
from scipy.interpolate import interp1d
import sys
import seaborn as sns
import pandas as pd
from matplotlib.collections import LineCollection
import ripser
from persim import plot_diagrams

import cebra
from cebra import CEBRA

# Function to write colored PLY file
def write_colored_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

def get_higher_percent_indices(trace_data, population_indices, discard_percent):
    # Calculate mean trace rate for the population
    population_trace = trace_data[:, population_indices]
    mean_trace_rate = np.mean(population_trace, axis=1)
    
    threshold = np.percentile(mean_trace_rate, discard_percent)
    
    # Get indices where mean trace rate is below the threshold
    higher_half = np.where(mean_trace_rate > threshold)[0]
    
    return higher_half


def find_nearest_time_index(spike_time, traj_time):
    return np.argmin(np.abs(np.array(traj_time) - spike_time))

def plot_barcode(topology_result, maxdim):
    fig, axs = plt.subplots(maxdim+1, 1, sharex=True, figsize=(7, 8))
    axs[0].set_xlim(0,2)
    cocycle = ["Points", "Loops", "Voids"]
    for k in range(maxdim+1):
        bars = topology_result['dgms'][k]
        bars[bars == np.inf] = 2
        lc = (
            np.vstack(
                [
                    bars[:, 0],
                    np.arange(len(bars), dtype=int) * 6,
                    bars[:, 1],
                    np.arange(len(bars), dtype=int) * 6,
                ]
            )
            .swapaxes(1, 0)
            .reshape(-1, 2, 2)
        )
        line_segments = LineCollection(lc, linewidth=5, color="gray", alpha=0.5)
        axs[k].set_ylabel(cocycle[k], fontsize=20)
        if k == 0:
            axs[k].set_ylim(len(bars) * 6 - 120, len(bars) * 6)
        elif k == 1:
            axs[k].set_ylim(0, len(bars) * 3 - 30)
        elif k == 2:
            axs[k].set_ylim(0, len(bars) * 6 + 10)
        axs[k].add_collection(line_segments)
        axs[k].set_yticks([])
        if k == 2:
            axs[k].set_xticks(np.linspace(0, 2, 3), np
                              .linspace(0, 2, 3), fontsize=15)
            axs[k].set_xlabel("Lifespan", fontsize=20)
    
    return fig 


def read_lifespan(ripser_output, dim):
    dim_diff = ripser_output['dgms'][dim][:, 1] - ripser_output['dgms'][dim][:, 0]
    if dim == 0:
        return dim_diff[~np.isinf(dim_diff)]
    else:
        return dim_diff

def get_max_lifespan(ripser_output_list, maxdim):
    lifespan_dic = {i: [] for i in range(maxdim+1)}
    # for f in ripser_output_list:
    f = ripser_output_list
    for dim in range(maxdim+1):
        lifespan = read_lifespan(f, dim)
        lifespan_dic[dim].extend(lifespan)
    return [max(lifespan_dic[i]) for i in range(maxdim+1)], lifespan_dic

def get_betti_number(ripser_output, shuffled_max_lifespan):
    bettis=[]
    for dim in range(len(ripser_output['dgms'])):
        lifespans=ripser_output['dgms'][dim][:, 1] - ripser_output['dgms'][dim][:, 0]
        betti_d = sum(lifespans > shuffled_max_lifespan[dim] * 1.1)
        bettis.append(betti_d)
    return bettis

def plot_lifespan(topology_dgms, shuffled_max_lifespan, ax, label_vis, maxdim):
    plot_diagrams(
        topology_dgms,
        ax=ax,
        legend=True,
    )

    ax.plot(
        [
            -0.5,
            2,
        ],
        [-0.5 + shuffled_max_lifespan[0], 2 + shuffled_max_lifespan[0]],
        color="C0",
        linewidth=3,
        alpha=0.5,

    )
    ax.plot(
        [
            -0.5,
            2,
        ],
        [-0.5 + shuffled_max_lifespan[1], 2 + shuffled_max_lifespan[1]],
        color="orange",
        linewidth=3,
        alpha=0.5,

    )
    if maxdim == 2:
        ax.plot(
            [-0.50, 2],
            [-0.5 + shuffled_max_lifespan[2], 2 + shuffled_max_lifespan[2]],
            color="green",
            linewidth=3,
            alpha=0.5,
        )
    ax.set_xlabel("Birth", fontsize=15)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([0, 1, 2])
    ax.tick_params(labelsize=13)
    if label_vis:
        ax.set_ylabel("Death", fontsize=15)
    else:
        ax.set_ylabel("")

def drawTopologyPV(cebra_posdir3, cebra_posdir8, cebra_posdir16, shuffled_cebra_posdir3, shuffled_cebra_posdir8, shuffled_cebra_posdir16, seed, maxdim, mouse_num, inactivation, namestring):
    np.random.seed(seed)
    random_idx=np.random.permutation(np.arange(len(cebra_posdir3)))[:2000]
    topology_dimension = {}
    for embedding in [cebra_posdir3, cebra_posdir8, cebra_posdir16]:
        ripser_output=ripser.ripser(embedding[random_idx], maxdim=maxdim, coeff=47)
        dimension = embedding.shape[1]
        topology_dimension[dimension] = ripser_output

    topology_random_dimension = {}
    for embedding in [shuffled_cebra_posdir3, shuffled_cebra_posdir8, shuffled_cebra_posdir16]:
        ripser_output=ripser.ripser(embedding[random_idx], maxdim=maxdim, coeff=47)
        dimension = embedding.shape[1]
        topology_random_dimension[dimension] = ripser_output

    for k in [3,8,16]:
        fig=plot_barcode(topology_dimension[k], maxdim)
        fig.suptitle(f'Dimension {k}')
        fig.show()

    fig = plt.figure(figsize=(18,5))

    for n, dim in enumerate([3,8,16]):
        shuffled_max_lifespan, lifespan_dic = get_max_lifespan(topology_random_dimension[dim], maxdim)
        ax = fig.add_subplot(1,3,n+1)
        ax.set_title(f'Dimension {dim}')
        plot_lifespan(topology_dimension[dim]['dgms'], shuffled_max_lifespan, ax, True, maxdim)
        print(f"Betti No. for dimension {dim}: {get_betti_number(topology_dimension[dim], shuffled_max_lifespan)}")

    fig.savefig(f"Figures-cebra/PV_Betti-{mouse_num}-{inactivation}-{namestring}")

def drawTopologyMEC(cebra_posdir3, cebra_posdir8, cebra_posdir16, shuffled_cebra_posdir3, shuffled_cebra_posdir8, shuffled_cebra_posdir16, seed, maxdim, inactivation, namestring):
    np.random.seed(seed)
    random_idx=np.random.permutation(np.arange(len(cebra_posdir3)))[:2000]
    topology_dimension = {}
    for embedding in [cebra_posdir3, cebra_posdir8, cebra_posdir16]:
        ripser_output=ripser.ripser(embedding[random_idx], maxdim=maxdim, coeff=47)
        dimension = embedding.shape[1]
        topology_dimension[dimension] = ripser_output

    topology_random_dimension = {}
    for embedding in [shuffled_cebra_posdir3, shuffled_cebra_posdir8, shuffled_cebra_posdir16]:
        ripser_output=ripser.ripser(embedding[random_idx], maxdim=maxdim, coeff=47)
        dimension = embedding.shape[1]
        topology_random_dimension[dimension] = ripser_output

    for k in [3,8,16]:
        fig=plot_barcode(topology_dimension[k], maxdim)
        fig.suptitle(f'Dimension {k}')
        fig.show()

    fig = plt.figure(figsize=(18,5))

    for n, dim in enumerate([3,8,16]):
        shuffled_max_lifespan, lifespan_dic = get_max_lifespan(topology_random_dimension[dim], maxdim)
        ax = fig.add_subplot(1,3,n+1)
        ax.set_title(f'Dimension {dim}')
        plot_lifespan(topology_dimension[dim]['dgms'], shuffled_max_lifespan, ax, True, maxdim)
        print(f"Betti No. for dimension {dim}: {get_betti_number(topology_dimension[dim], shuffled_max_lifespan)}")

    fig.savefig(f"Figures-cebra/MEC_Betti-{inactivation}-{namestring}")