import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from trajectory import rat_trajectory, rat_trajectory_mec, rat_trajectory_PV
from scipy.interpolate import interp1d
import pandas as pd
from utils import *
from downsampling import *

import cebra
from cebra import CEBRA

#Parameters
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="PV trace cebra parameters")

# Add arguments for each parameter
parser.add_argument('--mouse_num', type=int, default=2, help="Number of mice")
parser.add_argument('--inactivation', type=lambda x: (str(x).lower() == 'true'), choices=[True, False], default=True, help="Inactivation flag")
parser.add_argument('--max_iterations', type=int, default=1000, help="Maximum number of iterations")
parser.add_argument('--maxdim', type=int, default=1, help="Maximum dimension of topology analysis")
parser.add_argument('--seed', type=int, default=111, help="Random seed")
parser.add_argument('--discard_low_fire', type=int, default=0, help="Percentage(max 100) to discard low firing rate timesteps")

# Parse the arguments
args = parser.parse_args()

# Use the parsed arguments
mouse_num = args.mouse_num
inactivation = args.inactivation
max_iterations = args.max_iterations
maxdim = args.maxdim
seed = args.seed
discard_percent = args.discard_low_fire
ply_name = f'PV_{mouse_num}_{inactivation}'

print(f"Starting trial {mouse_num}-{inactivation}.")

#1
# index1 = [2, 3, 5, 6, 10, 13, 14, 17, 19, 21, 26, 32, 33, 38, 40, 43, 44, 47, 48, 54, 57, 63, 71, 73, 74]
# index2 = [9, 16, 18, 20, 22, 27, 36, 39, 45, 52, 53, 58, 69]

#2
index1 = [3, 10, 11, 25, 26, 27, 31, 35, 40, 42, 49, 50, 59, 61, 62, 64, 65, 68, 77, 78, 80, 83, 86, 91, 92, 93]
index2 = [14, 29, 30, 34, 39, 41, 44, 45, 48, 60, 71, 73, 76, 81, 82, 84, 87, 88, 89] #False

#3
# index1 = [0, 2, 4, 8, 9, 11, 14, 17, 19, 25, 26, 27, 28, 30, 34, 35, 36, 38, 42, 46, 48, 52, 53, 56, 61, 65, 69, 71, 73, 74, 77, 78, 79, 81, 82, 83, 84, 86, 91, 93, 96, 98, 99, 100, 101, 105, 106, 107]
# index2 = [5, 10, 12, 15, 22, 24, 32, 39, 40, 43, 44, 54, 55, 57, 58, 59, 60, 62, 64, 67, 70, 85, 87, 88, 92, 95, 102, 104]

if mouse_num == 1 and not inactivation:
    spk_filename = f'data/PV_inactivation/JY/mouse1 - base and opto/Square50_B1O1B2O2_PV5_6_Base1_deconv.xlsx'
    trc_filename = f'data/PV_inactivation/JY/mouse1 - base and opto/Square50_B1O1B2O2_PV5_6_Base1.xlsx'
elif mouse_num == 1 and inactivation:
    spk_filename = f'data/PV_inactivation/JY/mouse1 - base and opto/Square50_B1O1B2O2_PV5_6_Opto1_deconv.xlsx'
    trc_filename = f'data/PV_inactivation/JY/mouse1 - base and opto/Square50_B1O1B2O2_PV5_6_Opto1.xlsx'
elif mouse_num == 2 and not inactivation:
    spk_filename = f'data/PV_inactivation/JY/mouse2 - base and opto/Square50_BO_PV5_7_Base_deconv.xlsx'
    trc_filename = f'data/PV_inactivation/JY/mouse2 - base and opto/Square50_BO_PV5_7_Base.xlsx'
elif mouse_num == 2 and inactivation:
    spk_filename = f'data/PV_inactivation/JY/mouse2 - base and opto/Square50_BO_PV5_7_Opto_deconv.xlsx'
    trc_filename = f'data/PV_inactivation/JY/mouse2 - base and opto/Square50_BO_PV5_7_Opto.xlsx'
elif mouse_num == 3 and not inactivation:
    spk_filename = f'data/PV_inactivation/JY/mouse3 - base and chemo/Square50_BCB12_PV6_2_2_Base_deconv.xlsx'
    trc_filename = f'data/PV_inactivation/JY/mouse3 - base and chemo/Square50_BCB12_PV6_2_2_Base.xlsx'
elif mouse_num == 3 and inactivation:
    spk_filename = f'data/PV_inactivation/JY/mouse3 - base and chemo/Square50_BCB12_PV6_2_2_Chemo_deconv.xlsx'
    trc_filename = f'data/PV_inactivation/JY/mouse3 - base and chemo/Square50_BCB12_PV6_2_2_Chemo.xlsx'

# Read the CSV file
df = pd.read_csv(trc_filename, header=None)

# Get cell numbers
cell_numbers = df.iloc[0].dropna().tolist()[1:]
trace_per_cell = {cell: [] for cell in cell_numbers}
spike_times = {cell: [] for cell in cell_numbers}
nn_trace_per_cell = {cell: [] for cell in cell_numbers}

# Get time data
time_data = [float(i) for i in df.iloc[1:, 1].tolist()]
start_time = time_data[0]
time_data = [time_data[i] - start_time for i in range(len(time_data))]

traj_time, x_pos, y_pos, head_dir, velocity = rat_trajectory_PV(inactivation, mouse_num)

# get trace data from each cell + interpolate
for i in range(len(cell_numbers)):
    cell_name = np.asarray(cell_numbers)[i]
    trace_data = [float(i) for i in df.iloc[1:, i+2].tolist()]
    time_interp = interp1d(time_data, trace_data, bounds_error=False, fill_value="extrapolate")

    trace_data = time_interp(traj_time)
    nn_trace_per_cell[cell_name] = trace_data.tolist()
    trace_data = (trace_data - np.mean(trace_data)) / np.std(trace_data)
    trace_per_cell[cell_name] = trace_data.tolist()

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

total_trace = np.zeros((len(traj_time), len(cell_numbers)))

# Iterate through cells and their spike times
for i, cell in enumerate(cell_numbers):
    total_trace[:, i] = np.asarray(trace_per_cell[cell])

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

else:
    total_trace1 = total_trace[:, :]
    total_trace2 = total_trace[:, :]

neural1 = total_trace1[:, index1] #index at 2
neural2 = total_trace2[:, index2] #index at 2

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
        seed, maxdim, mouse_num, inactivation, "1"
    ),
    (
        downsampled_arrays['cebra_posdir3_2'],
        downsampled_arrays['cebra_posdir8_2'],
        downsampled_arrays['cebra_posdir16_2'],
        adjusted_shuffled_arrays['shuffled_cebra_posdir3_2'],
        adjusted_shuffled_arrays['shuffled_cebra_posdir8_2'],
        adjusted_shuffled_arrays['shuffled_cebra_posdir16_2'],
        seed, maxdim, mouse_num, inactivation, "2"
    ),
    (
        np.concatenate([downsampled_arrays['cebra_posdir3_1'], downsampled_arrays['cebra_posdir3_2']]),
        np.concatenate([downsampled_arrays['cebra_posdir8_1'], downsampled_arrays['cebra_posdir8_2']]),
        np.concatenate([downsampled_arrays['cebra_posdir16_1'], downsampled_arrays['cebra_posdir16_2']]),
        np.concatenate([adjusted_shuffled_arrays['shuffled_cebra_posdir3_1'], adjusted_shuffled_arrays['shuffled_cebra_posdir3_2']]),
        np.concatenate([adjusted_shuffled_arrays['shuffled_cebra_posdir8_1'], adjusted_shuffled_arrays['shuffled_cebra_posdir8_2']]),
        np.concatenate([adjusted_shuffled_arrays['shuffled_cebra_posdir16_1'], adjusted_shuffled_arrays['shuffled_cebra_posdir16_2']]),
        seed, maxdim, mouse_num, inactivation, "12"
    )
]

for args in topology_calls:
    drawTopologyPV(*args)