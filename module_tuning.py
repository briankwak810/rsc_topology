import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
from trajectory import rat_trajectory

def plot_average_firing_rate(shape, cell_indices):
    # Parameters
    spk_filename = f'data/B6_8_1_{shape}_Spike.csv'
    trc_filename = f'data/B6_8_1_{shape}_Trace.csv'

    # Read the CSV file
    df = pd.read_csv(trc_filename, header=None)

    # Get cell numbers
    cell_numbers = df.iloc[0].dropna().tolist()[1:]
    spike_times = {cell: [] for cell in cell_numbers}

    # Get time data
    time_data = [float(i) for i in df.iloc[2:, 0].tolist()]
    start_time = time_data[0]
    time_data = [time_data[i] - start_time for i in range(len(time_data))]

    with open(spk_filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            time = float(row[0]) - start_time
            cell = row[1]
            if cell in spike_times:
                spike_times[cell].append(time)

    traj_time, x_pos, y_pos, head_dir, velocity = rat_trajectory(shape)

    # NaN interpolation
    for pos in [x_pos, y_pos, head_dir]:
        invalid = np.isnan(pos)
        pos_interp = pos.copy()
        valid = np.where(~invalid)[0]
        pos_interp[invalid] = np.interp(np.where(invalid)[0], valid, pos[valid])
        pos[:] = pos_interp

    # Discard index where velocity < threshold
    threshold = 2
    moving_idx = velocity > threshold

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Prepare for rate map calculation
    num_bins = 50
    position_hist, x_edges, y_edges = np.histogram2d(x_pos[moving_idx], y_pos[moving_idx], bins=num_bins)

    # Calculate average firing map
    average_firing_map = np.zeros_like(position_hist)

    for cell_index in cell_indices:
        cell_name = cell_numbers[cell_index]
        fire_xpos = []
        fire_ypos = []
        
        for spike_time in spike_times[cell_name]:
            closest_index = np.argmin(np.abs(np.array(traj_time) - spike_time))
            if moving_idx[closest_index]:
                fire_xpos.append(x_pos[closest_index])
                fire_ypos.append(y_pos[closest_index])

        spike_hist, _, _ = np.histogram2d(fire_xpos, fire_ypos, bins=[x_edges, y_edges])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            firing_map = np.divide(spike_hist, position_hist)
            firing_map[np.isinf(firing_map)] = 0
            firing_map[np.isnan(firing_map)] = 0
        
        average_firing_map += firing_map

    average_firing_map /= len(cell_indices)

    # Smooth the average firing map
    sigma = 1.3
    average_firing_map_smooth = gaussian_filter(average_firing_map, sigma)
    average_firing_map_smooth = np.ma.array(average_firing_map_smooth.T)

    # Plot the average rate map
    im = ax.imshow(average_firing_map_smooth, origin='lower', 
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   cmap='viridis', interpolation='bilinear')

    ax.plot(x_pos, y_pos, color='gray', alpha=0.5, linewidth=0.5)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Average Firing Rate')

    ax.set_title(f'Average Firing Map for Specified Population in {shape}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

# Example usage:
shape = 'Sqr'
cell_indices = [0, 5, 6, 8, 11, 16, 17, 26, 27, 28, 34, 35, 39, 49, 57]
plot_average_firing_rate(shape, cell_indices)