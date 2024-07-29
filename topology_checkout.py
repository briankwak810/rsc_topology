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

################# TOPOLOGY ####################

downsampled_arrays = np.load('downsampled_embedding_arrays.npz')
adjusted_shuffled_arrays = np.load('adjusted_shuffled_embedding_arrays.npz')


embedding_arrays = np.load('embedding_arrays.npz')
shuffled_arrays = np.load('shuffled_embedding_arrays.npz')

# # Process each array pair in the npz files
# downsampled_arrays = {}
# adjusted_shuffled_arrays = {}

# for name, array in embedding_arrays.items():
#     print(name)
#     shuffled_name = f"shuffled_{name}"
#     if shuffled_name in shuffled_arrays:
#         downsampled, adjusted_shuffled = process_array(array, shuffled_arrays[shuffled_name], name)
#         downsampled_arrays[name] = downsampled
#         adjusted_shuffled_arrays[shuffled_name] = adjusted_shuffled
#     else:
#         print(f"Warning: No corresponding shuffled array found for {name}")

# # Save the downsampled and adjusted shuffled arrays
# np.savez('downsampled_embedding_arrays.npz', **downsampled_arrays)
# np.savez('adjusted_shuffled_embedding_arrays.npz', **adjusted_shuffled_arrays)
# print("\nAll downsampled and adjusted shuffled arrays saved successfully.")

# # Print summary
# print("\nSummary:")
# for name, array in downsampled_arrays.items():
#     original_size = len(embedding_arrays[name])
#     downsampled_size = len(array)
#     shuffled_size = len(adjusted_shuffled_arrays[f"shuffled_{name}"])
#     percentage = downsampled_size / original_size
#     print(f"{name}: Original {original_size}, Downsampled {downsampled_size}, Adjusted Shuffled {shuffled_size} ({percentage:.2%})")

# # Set up function calls for topology calculation
# topology_calls = [
#     (
#         downsampled_arrays['cebra_posdir3_1'],
#         downsampled_arrays['cebra_posdir8_1'],
#         downsampled_arrays['cebra_posdir16_1'],
#         adjusted_shuffled_arrays['shuffled_cebra_posdir3_1'],
#         adjusted_shuffled_arrays['shuffled_cebra_posdir8_1'],
#         adjusted_shuffled_arrays['shuffled_cebra_posdir16_1'],
#         seed, maxdim, mouse_num, inactivation, "1"
#     ),
#     (
#         downsampled_arrays['cebra_posdir3_2'],
#         downsampled_arrays['cebra_posdir8_2'],
#         downsampled_arrays['cebra_posdir16_2'],
#         adjusted_shuffled_arrays['shuffled_cebra_posdir3_2'],
#         adjusted_shuffled_arrays['shuffled_cebra_posdir8_2'],
#         adjusted_shuffled_arrays['shuffled_cebra_posdir16_2'],
#         seed, maxdim, mouse_num, inactivation, "2"
#     ),
#     (
#         np.concatenate([downsampled_arrays['cebra_posdir3_1'], downsampled_arrays['cebra_posdir3_2']]),
#         np.concatenate([downsampled_arrays['cebra_posdir8_1'], downsampled_arrays['cebra_posdir8_2']]),
#         np.concatenate([downsampled_arrays['cebra_posdir16_1'], downsampled_arrays['cebra_posdir16_2']]),
#         np.concatenate([adjusted_shuffled_arrays['shuffled_cebra_posdir3_1'], adjusted_shuffled_arrays['shuffled_cebra_posdir3_2']]),
#         np.concatenate([adjusted_shuffled_arrays['shuffled_cebra_posdir8_1'], adjusted_shuffled_arrays['shuffled_cebra_posdir8_2']]),
#         np.concatenate([adjusted_shuffled_arrays['shuffled_cebra_posdir16_1'], adjusted_shuffled_arrays['shuffled_cebra_posdir16_2']]),
#         seed, maxdim, mouse_num, inactivation, "12"
#     )
# ]

# for args in topology_calls:
#     drawTopologyPV(*args)

'''
cebra_posdir8_1 = embedding_arrays['cebra_po''sdir16_1']
cebra_posdir8_2 = embedding_arrays['cebra_posdir16_2']

downsampled_8_1 = downsampled_arrays['cebra_posdir16_1']
downsampled_8_2 = downsampled_arrays['cebra_posdir16_2']


blue = np.array([0, 0, 1])  # RGB for blue
orange = np.array([1, 0.5, 0])  # RGB for orange

# Create color array
colors = np.zeros((len(cebra_posdir8_1) + len(cebra_posdir8_2), 3))
colors[:len(cebra_posdir8_1)] = blue
colors[len(cebra_posdir8_2):] = orange

colors_uint8 = (colors * 255).astype(np.uint8)

# Create color array
colors = np.zeros((len(downsampled_8_1) + len(downsampled_8_2), 3))
colors[:len(downsampled_8_1)] = blue
colors[len(downsampled_8_2):] = orange

downsampled_colors_uint8 = (colors * 255).astype(np.uint8)

# Write the colored PLY file
write_colored_ply(f"original.ply", np.concatenate([cebra_posdir8_1, cebra_posdir8_2]), colors_uint8)
write_colored_ply(f"downsampled.ply", np.concatenate([downsampled_8_1, downsampled_8_2]), downsampled_colors_uint8)

print("Embedding to ply complete.")'''

cebra_posdir8_1 = embedding_arrays['cebra_posdir16_1']
downsampled_8_1 = downsampled_arrays['cebra_posdir16_1']
blue = np.array([0, 0, 1])  # RGB for blue

# Create color array for original data
colors = np.zeros((len(cebra_posdir8_1), 3))
colors[:] = blue
colors_uint8 = (colors * 255).astype(np.uint8)

# Create color array for downsampled data
downsampled_colors = np.zeros((len(downsampled_8_1), 3))
downsampled_colors[:] = blue
downsampled_colors_uint8 = (downsampled_colors * 255).astype(np.uint8)

# Write the colored PLY files
write_colored_ply("original_blue.ply", cebra_posdir8_1, colors_uint8)
write_colored_ply("downsampled_blue.ply", downsampled_8_1, downsampled_colors_uint8)

print("Embedding of blue object to ply complete.")