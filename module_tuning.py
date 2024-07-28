import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
from trajectory import rat_trajectory, rat_trajectory_mec, rat_trajectory_PV

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

def plot_average_firing_rate_mec(cell_indices, inactivation):
    # Parameters
    spk_filename = f'data/mec_inactivation/MEC_inactivation/Spike.csv'
    trc_filename = f'data/mec_inactivation/MEC_inactivation/Trace.csv'

    # Read the CSV file
    df = pd.read_csv(trc_filename, header=None)

    # Get cell numbers
    cell_numbers = df.iloc[0].dropna().tolist()[1:]
    spike_times = {cell: [] for cell in cell_numbers}

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

    with open(spk_filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            time = float(row[0])
            cell = row[1]
            if inactivation and time > 2000:
                spike_times[cell].append(time - opto_start)
            elif not inactivation and time < 2000:
                spike_times[cell].append(time)

    traj_time, x_pos, y_pos, head_dir, velocity = rat_trajectory_mec(inactivation)

    # NaN interpolation
    for pos in [x_pos, y_pos, head_dir, velocity]:
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

    ax.set_title(f'Average Firing Map for Specified Population in MEC-{inactivation}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

def plot_average_firing_rate_PV(inactivation, mouse_num):
    # Parameters
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

    # Read the excel file
    df = pd.read_csv(trc_filename, header=None)

    # Get cell numbers
    cell_numbers = df.iloc[0].dropna().tolist()[1:]
    spike_times = {cell: [] for cell in cell_numbers}

    # Get time data
    time_data = [float(i) for i in df.iloc[1:, 1].tolist()]
    start_time = time_data[0]
    time_data = [time_data[i] - start_time for i in range(len(time_data))]

    with open(spk_filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) # Skip the header row
        for row in csv_reader:
            time = float(row[1]) - start_time
            cell = row[2]
            spike_times[cell].append(time)

    traj_time, x_pos, y_pos, head_dir, velocity = rat_trajectory_PV(inactivation, mouse_num)

    # NaN interpolation
    for pos in [x_pos, y_pos, head_dir, velocity]:
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

    ax.set_title(f'Average Firing Map for Specified Population in PV-{mouse_num}-{inactivation}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

# Example usage:

cell_indices = [3, 4, 6, 7, 9, 14, 15, 21, 22]
plot_average_firing_rate_mec(cell_indices, False)