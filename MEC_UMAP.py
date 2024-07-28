import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from trajectory import rat_trajectory, rat_trajectory_mec, rat_trajectory_PV
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from utils import *
import umap

import cebra
from cebra import CEBRA

#Parameters
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="Your script description")

# Add arguments for each parameter
parser.add_argument('--inactivation', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default=True, help="Inactivation flag")
parser.add_argument('--usespikes', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default=False, help="Use spikes flag")
parser.add_argument('--dim', type=int, default=8, help="UMAP embedding dimension")
parser.add_argument('--discard_low_fire', type=int, default=0, help="Percentage(max 100) to discard low firing rate timesteps")

# Parse the arguments
args = parser.parse_args()

# Use the parsed arguments
inactivation = args.inactivation
usespikes = args.usespikes
ply_name = f'MEC_UMAP_{inactivation}_spk' if usespikes else f'MEC_UMAP_{inactivation}'
dim = args.dim
discard_percent = args.discard_low_fire

print(f"Starting trial {ply_name}.")

# Spk index
# index1 = [0, 2, 5, 10, 11, 18, 20, 23, 24, 25, 26, 27, 30, 35, 39, 40, 41]
# index2 = [12, 13, 16, 17, 28, 29, 31, 32, 33, 36, 37, 38, 42, 44]

# Trc index
index1 = [0, 9, 10, 11, 18, 20, 21, 23, 24, 25, 26, 27, 35, 36, 39, 40, 41, 43, 46]
index2 = [1, 8, 12, 14, 15, 16, 17, 28, 29, 31, 32, 33, 34, 37, 42, 44, 45]

spk_filename = f'data/mec_inactivation/MEC_inactivation/Spike.csv'
trc_filename = f'data/mec_inactivation/MEC_inactivation/Trace.csv'

# Read the CSV file
df = pd.read_csv(trc_filename, header=None)

# Get cell numbers
cell_numbers = df.iloc[0].dropna().tolist()[1:]
trace_per_cell = {cell: [] for cell in cell_numbers}
spike_times = {cell: [] for cell in cell_numbers}
nn_trace_per_cell = {cell: [] for cell in cell_numbers}

# Get time data
time_data = [float(i) for i in df.iloc[2:, 0].tolist()]
start_time = time_data[0]
time_data = [time_data[i] - start_time for i in range(len(time_data))]

if inactivation:
    time_data = time_data[54055:]
    opto_start = time_data[0]
    time_data = [time - time_data[0] for time in time_data]
else:
    time_data = time_data[:54054]

traj_time, x_pos, y_pos, head_dir, velocity = rat_trajectory_mec(inactivation)

# NaN interpolation
for pos in [x_pos, y_pos, head_dir, velocity]:
    invalid = np.isnan(pos)
    pos_interp = pos.copy()
    valid = np.where(~invalid)[0]
    pos_interp[invalid] = np.interp(np.where(invalid)[0], valid, pos[valid])
    pos[:] = pos_interp

# discard index where velocity < threshold
threshold = 2
moving_idx = [i for i, x in enumerate(velocity) if x > threshold]

# get trace data from each cell + interpolate
for i in range(np.shape(df.iloc[2:])[1] - 1):
    cell_name = np.asarray(cell_numbers)[i]
    all_trace_data = [float(i) for i in df.iloc[2:, i+1].tolist()]
    if inactivation:
        trace_data = all_trace_data[54055:]
    else:
        trace_data = all_trace_data[:54054]
    time_interp = interp1d(time_data, trace_data, bounds_error=False, fill_value="extrapolate")

    trace_data = time_interp(traj_time)
    nn_trace_per_cell[cell_name] = trace_data.tolist()
    trace_data = (trace_data - np.mean(trace_data)) / np.std(trace_data)
    trace_per_cell[cell_name] = trace_data.tolist()

# get spike data from each cell + interpolate
with open(spk_filename, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        time = float(row[0])
        cell = row[1]
        if inactivation and (time > 2000):
            spike_times[cell].append(time - opto_start)
        elif not inactivation and (time < 2000):
            spike_times[cell].append(time)

total_trace = np.zeros((len(traj_time), len(cell_numbers)))

# Iterate through cells and their spike times
for i, cell in enumerate(cell_numbers):
    total_trace[:, i] = np.asarray(trace_per_cell[cell])

# Get spike data -> nearest interpolation
spiking_data = np.zeros((len(traj_time), len(cell_numbers)))

for i, cell in enumerate(cell_numbers):
    for spike_time in spike_times[cell]:
        nearest_index = find_nearest_time_index(spike_time, traj_time)
        if nearest_index in moving_idx:
            spiking_data[nearest_index, i] = 1

# Apply Gaussian filter
sigma_ms = 0.2  # sigma in milliseconds
sigma_samples = sigma_ms / (traj_time[1] - traj_time[0])  # convert sigma to samples
spiking_data = gaussian_filter1d(spiking_data, sigma=sigma_samples, axis=0)

if discard_percent != 0:
    # Not-Normalized total trace to discard lower percent
    nn_total_trace = np.zeros((len(traj_time), len(cell_numbers)))
    for i, cell in enumerate(cell_numbers):
        nn_total_trace[:, i] = np.asarray(nn_trace_per_cell[cell])

    # Get population
    higher_pop1 = get_higher_percent_indices(nn_total_trace, index1, discard_percent)
    higher_pop2 = get_higher_percent_indices(nn_total_trace, index2, discard_percent)

    total_trace1 = total_trace[higher_pop1, :]
    total_trace2 = total_trace[higher_pop2, :]

    total_spk1 = spiking_data[higher_pop1, :]
    total_spk2 = spiking_data[higher_pop2, :]

else:
    total_trace1 = total_trace[:, :]
    total_trace2 = total_trace[:, :]
    total_spk1 = spiking_data
    total_spk2 = spiking_data

if not usespikes:
    neural1 = total_trace1[:, index1] #index at 2
    neural2 = total_trace2[:, index2] #index at 2
else:
    neural1 = total_spk1
    neural2 = total_spk2

# normalize (for square)
x_pos = x_pos / 10
y_pos = y_pos / 10

min_x = np.min(x_pos)
max_x = np.max(x_pos)
min_y = np.min(y_pos)
max_y = np.max(y_pos)

continuous_index = np.column_stack((x_pos-min_x, max_x-x_pos, y_pos-min_y, max_y-y_pos))

if discard_percent != 0:
    continuous_index1 = continuous_index[higher_pop1, :]
    continuous_index2 = continuous_index[higher_pop2, :]
else:
    continuous_index1 = continuous_index
    continuous_index2 = continuous_index

print("Done data cleanup... Starting embedding 1.")

############## UMAP ##############

umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=1,
    n_components=dim,  # or whatever dimensionality you want
    metric='euclidean',
    random_state=42
)

embeddings1 = umap_model.fit_transform(np.hstack([neural1, continuous_index1]))

print("Done embedding 1... Starting embedding 2.")

embeddings2 = umap_model.fit_transform(np.hstack([neural2, continuous_index2]))

print("Done embedding 2.")

blue = np.array([0, 0, 1])  # RGB for blue
orange = np.array([1, 0.5, 0])  # RGB for orange

# Create color array
colors = np.zeros((len(embeddings1) + len(embeddings2), 3))
colors[:len(embeddings1)] = blue
colors[len(embeddings2):] = orange

colors = (colors * 255).astype(np.uint8)

# Write the PLY file
write_colored_ply(f"embedding_3d_structures/colored_points_{ply_name}_{dim}.ply", np.concatenate([embeddings1, embeddings2]), colors)