import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from trajectory import rat_trajectory, rat_trajectory_mec, rat_trajectory_PV
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from utils import *
from downsampling import *

import cebra
from cebra import CEBRA

#Parameters
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="Your script description")

# Add arguments for each parameter
parser.add_argument('--inactivation', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default=True, help="Inactivation flag")
parser.add_argument('--max_iterations', type=int, default=1000, help="Maximum number of iterations")
parser.add_argument('--maxdim', type=int, default=1, help="Maximum dimension of topology analysis")
parser.add_argument('--seed', type=int, default=111, help="Random seed")
parser.add_argument('--discard_low_fire', type=int, default=0, help="Percentage(max 100) to discard low firing rate timesteps")
parser.add_argument('--usespikes', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default=False, help="Use spikes flag")

# Parse the arguments
args = parser.parse_args()

# Use the parsed arguments
inactivation = args.inactivation
max_iterations = args.max_iterations
maxdim = args.maxdim
seed = args.seed
discard_percent = args.discard_low_fire
usespikes = args.usespikes
ply_name = f'MEC_{inactivation}_spk' if usespikes else f'MEC_{inactivation}'

print(f"Starting trial {ply_name}.")

# Spk index
index1 = [0, 2, 5, 10, 11, 18, 20, 23, 24, 25, 26, 27, 30, 35, 39, 40, 41]
index2 = [12, 13, 16, 17, 28, 29, 31, 32, 33, 36, 37, 38, 42, 44]

# Trc index
# index1 = [0, 9, 10, 11, 18, 20, 21, 23, 24, 25, 26, 27, 35, 36, 39, 40, 41, 43, 46]
# index2 = [1, 8, 12, 14, 15, 16, 17, 28, 29, 31, 32, 33, 34, 37, 42, 44, 45]

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

############## CEBRA training ##############

cebra_posdir3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature_mode = 'constant',
                        temperature=1.5,
                        min_temperature = 1e-1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        hybrid=False)

cebra_posdir8_model= CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature_mode = 'constant',
                        temperature=1.5,
                        min_temperature = 1e-1,
                        output_dimension=8,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        hybrid=False)

cebra_posdir16_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature_mode = 'constant',
                        temperature=1.5,
                        min_temperature = 1e-1,
                        output_dimension=16,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        hybrid=False)

cebra_posdir3_model.fit(neural1, continuous_index1)
cebra_posdir8_model.fit(neural1, continuous_index1)
cebra_posdir16_model.fit(neural1, continuous_index1)

cebra_posdir3_1 = cebra_posdir3_model.transform(neural1)
cebra_posdir8_1 = cebra_posdir8_model.transform(neural1)
cebra_posdir16_1 = cebra_posdir16_model.transform(neural1)

print("CEBRA embedding for nerual1 complete.")

cebra_posdir3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature_mode = 'constant',
                        temperature=1.5,
                        min_temperature = 1e-1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        hybrid=False)

cebra_posdir8_model= CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature_mode = 'constant',
                        temperature=1.5,
                        min_temperature = 1e-1,
                        output_dimension=8,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        hybrid=False)

cebra_posdir16_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature_mode = 'constant',
                        temperature=1.5,
                        min_temperature = 1e-1,
                        output_dimension=16,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        hybrid=False)

cebra_posdir3_model.fit(neural2, continuous_index2)

cebra_posdir8_model.fit(neural2, continuous_index2)

cebra_posdir16_model.fit(neural2, continuous_index2)

cebra_posdir3_2 = cebra_posdir3_model.transform(neural2)
cebra_posdir8_2 = cebra_posdir8_model.transform(neural2)
cebra_posdir16_2 = cebra_posdir16_model.transform(neural2)

print("CEBRA embedding for nerual2 complete.")

np.savez('embedding_arrays.npz', 
         cebra_posdir3_1=cebra_posdir3_1,
         cebra_posdir3_2=cebra_posdir3_2,
         cebra_posdir8_1=cebra_posdir8_1,
         cebra_posdir8_2=cebra_posdir8_2,
         cebra_posdir16_1=cebra_posdir16_1,
         cebra_posdir16_2=cebra_posdir16_2)

print("All 6 arrays saved successfully.")

############## VISUALIZE to PLY file ##############

## For other visualization ##

# embedding_labels = head_dir
# norm = plt.Normalize(embedding_labels.min(), embedding_labels.max())
# normalized_index = norm(embedding_labels)

# cmap = plt.get_cmap('plasma')  # or 'viridis'
# colors_float = cmap(normalized_index)[:, :3]

# colors = (colors_float * 255).astype(int)
# colors1 = colors[higher_pop1]
# colors2 = colors[higher_pop2]
# colors_uint8 = np.concatenate([colors1, colors2])

blue = np.array([0, 0, 1])  # RGB for blue
orange = np.array([1, 0.5, 0])  # RGB for orange

# Create color array
colors = np.zeros((len(cebra_posdir3_1) + len(cebra_posdir3_2), 3))
colors[:len(cebra_posdir3_1)] = blue
colors[len(cebra_posdir3_2):] = orange

colors_uint8 = (colors * 255).astype(np.uint8)

# Write the colored PLY file
dim = 3
write_colored_ply(f"embedding_3d_structures/colored_points_{ply_name}_{dim}.ply", np.concatenate([cebra_posdir3_1, cebra_posdir3_2]), colors_uint8)
dim = 8
write_colored_ply(f"embedding_3d_structures/colored_points_{ply_name}_{dim}.ply", np.concatenate([cebra_posdir8_1, cebra_posdir8_2]), colors_uint8)
dim = 16
write_colored_ply(f"embedding_3d_structures/colored_points_{ply_name}_{dim}.ply", np.concatenate([cebra_posdir16_1, cebra_posdir16_2]), colors_uint8)

print("Embedding to ply complete.")

########### SHUFFLE for TOPOLOGY ############

shuffled_index = np.random.permutation(continuous_index1)

shuffled_cebra_posdir3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1.5,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

shuffled_cebra_posdir8_model= CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1.5,
                        output_dimension=8,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

shuffled_cebra_posdir16_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1.5,
                        output_dimension=16,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

shuffled_cebra_posdir3_model.fit(neural1, shuffled_index)
shuffled_cebra_posdir8_model.fit(neural1, shuffled_index)
shuffled_cebra_posdir16_model.fit(neural1, shuffled_index)

shuffled_cebra_posdir3_1 = shuffled_cebra_posdir3_model.transform(neural1)
shuffled_cebra_posdir8_1 = shuffled_cebra_posdir8_model.transform(neural1)
shuffled_cebra_posdir16_1 = shuffled_cebra_posdir16_model.transform(neural1)

shuffled_index = np.random.permutation(continuous_index2)

shuffled_cebra_posdir3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1.5,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

shuffled_cebra_posdir8_model= CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1.5,
                        output_dimension=8,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

shuffled_cebra_posdir16_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1.5,
                        output_dimension=16,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

shuffled_cebra_posdir3_model.fit(neural2, shuffled_index)
shuffled_cebra_posdir8_model.fit(neural2, shuffled_index)
shuffled_cebra_posdir16_model.fit(neural2, shuffled_index)

shuffled_cebra_posdir3_2 = shuffled_cebra_posdir3_model.transform(neural2)
shuffled_cebra_posdir8_2 = shuffled_cebra_posdir8_model.transform(neural2)
shuffled_cebra_posdir16_2 = shuffled_cebra_posdir16_model.transform(neural2)

cebra_posdir3 = np.concatenate([cebra_posdir3_1, cebra_posdir3_2])
cebra_posdir8 = np.concatenate([cebra_posdir8_1, cebra_posdir8_2])
cebra_posdir16 = np.concatenate([cebra_posdir16_1, cebra_posdir16_2])

shuffled_cebra_posdir3 = np.concatenate([shuffled_cebra_posdir3_1, shuffled_cebra_posdir3_2])
shuffled_cebra_posdir8 = np.concatenate([shuffled_cebra_posdir8_1, shuffled_cebra_posdir8_2])
shuffled_cebra_posdir16 = np.concatenate([shuffled_cebra_posdir16_1, shuffled_cebra_posdir16_2])

np.savez('shuffled_embedding_arrays.npz',
         shuffled_cebra_posdir3_1=shuffled_cebra_posdir3_1,
         shuffled_cebra_posdir3_2=shuffled_cebra_posdir3_2,
         shuffled_cebra_posdir8_1=shuffled_cebra_posdir8_1,
         shuffled_cebra_posdir8_2=shuffled_cebra_posdir8_2,
         shuffled_cebra_posdir16_1=shuffled_cebra_posdir16_1,
         shuffled_cebra_posdir16_2=shuffled_cebra_posdir16_2)

print("All 6 arrays saved independently.")

################# TOPOLOGY ####################

embedding_arrays = np.load('embedding_arrays.npz')
shuffled_arrays = np.load('shuffled_embedding_arrays.npz')

# Process each array pair in the npz files
downsampled_arrays = {}
adjusted_shuffled_arrays = {}

for name, array in embedding_arrays.items():
    shuffled_name = f"shuffled_{name}"
    if shuffled_name in shuffled_arrays:
        downsampled, adjusted_shuffled = process_array(array, shuffled_arrays[shuffled_name], name)
        downsampled_arrays[name] = downsampled
        adjusted_shuffled_arrays[shuffled_name] = adjusted_shuffled
    else:
        print(f"Warning: No corresponding shuffled array found for {name}")

# Save the downsampled and adjusted shuffled arrays
np.savez('downsampled_embedding_arrays.npz', **downsampled_arrays)
np.savez('adjusted_shuffled_embedding_arrays.npz', **adjusted_shuffled_arrays)
print("\nAll downsampled and adjusted shuffled arrays saved successfully.")

# Print summary
print("\nSummary:")
for name, array in downsampled_arrays.items():
    original_size = len(embedding_arrays[name])
    downsampled_size = len(array)
    shuffled_size = len(adjusted_shuffled_arrays[f"shuffled_{name}"])
    percentage = downsampled_size / original_size
    print(f"{name}: Original {original_size}, Downsampled {downsampled_size}, Adjusted Shuffled {shuffled_size} ({percentage:.2%})")


# Set up function calls for topology calculation
topology_calls = [
    (
        downsampled_arrays['cebra_posdir3_1'],
        downsampled_arrays['cebra_posdir8_1'],
        downsampled_arrays['cebra_posdir16_1'],
        adjusted_shuffled_arrays['shuffled_cebra_posdir3_1'],
        adjusted_shuffled_arrays['shuffled_cebra_posdir8_1'],
        adjusted_shuffled_arrays['shuffled_cebra_posdir16_1'],
        seed, maxdim, inactivation, "1"
    ),
    (
        downsampled_arrays['cebra_posdir3_2'],
        downsampled_arrays['cebra_posdir8_2'],
        downsampled_arrays['cebra_posdir16_2'],
        adjusted_shuffled_arrays['shuffled_cebra_posdir3_2'],
        adjusted_shuffled_arrays['shuffled_cebra_posdir8_2'],
        adjusted_shuffled_arrays['shuffled_cebra_posdir16_2'],
        seed, maxdim, inactivation, "2"
    ),
    (
        np.concatenate([downsampled_arrays['cebra_posdir3_1'], downsampled_arrays['cebra_posdir3_2']]),
        np.concatenate([downsampled_arrays['cebra_posdir8_1'], downsampled_arrays['cebra_posdir8_2']]),
        np.concatenate([downsampled_arrays['cebra_posdir16_1'], downsampled_arrays['cebra_posdir16_2']]),
        np.concatenate([adjusted_shuffled_arrays['shuffled_cebra_posdir3_1'], adjusted_shuffled_arrays['shuffled_cebra_posdir3_2']]),
        np.concatenate([adjusted_shuffled_arrays['shuffled_cebra_posdir8_1'], adjusted_shuffled_arrays['shuffled_cebra_posdir8_2']]),
        np.concatenate([adjusted_shuffled_arrays['shuffled_cebra_posdir16_1'], adjusted_shuffled_arrays['shuffled_cebra_posdir16_2']]),
        seed, maxdim, inactivation, "12"
    )
]

for args in topology_calls:
    drawTopologyMEC(*args)