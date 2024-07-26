import pandas as pd
import openpyxl
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import binned_statistic_2d
from trajectory import rat_trajectory, rat_trajectory_mec, rat_trajectory_PV
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize

########## CURRENTLY PV VERSION! ##############
#Perameters
inactivation = True
mouse_num = 3
cell_num = 1

cell_name = f' C{cell_num:02}'

# for shape in ['Tri', 'Sqr', 'Hex']: ###########
# spk_filename = f'data/B6_8_1_{shape}_Spike.csv'
# trc_filename = f'data/B6_8_1_{shape}_Trace.csv'

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

## NaN interpolation ##
# For x_pos
invalid_x_pos = np.isnan(x_pos)
x_pos_interp = x_pos.copy()
x_valid = np.where(~invalid_x_pos)[0]
x_pos_interp[invalid_x_pos] = np.interp(np.where(invalid_x_pos)[0], x_valid, x_pos[x_valid])

# For y_pos
invalid_y_pos = np.isnan(y_pos)
y_pos_interp = y_pos.copy()
y_valid = np.where(~invalid_y_pos)[0]
y_pos_interp[invalid_y_pos] = np.interp(np.where(invalid_y_pos)[0], y_valid, y_pos[y_valid])

# For HD
invalid_hd = np.isnan(head_dir)
hd_interp = head_dir.copy()
hd_valid = np.where(~invalid_hd)[0]
hd_interp[invalid_hd] = np.interp(np.where(invalid_hd)[0], hd_valid, head_dir[hd_valid])

# For vel
invalide_vel = np.isnan(velocity)
vel_interp = velocity.copy()
vel_valid = np.where(~invalide_vel)[0]
vel_interp[invalide_vel] = np.interp(np.where(invalide_vel)[0], vel_valid, velocity[vel_valid])

# Update the original arrays
x_pos = x_pos_interp
y_pos = y_pos_interp
head_dir = hd_interp
velocity = vel_interp

# discard index where velocity < threshold
threshold = 2
moving_idx = [i for i, x in enumerate(velocity) if x > threshold]

for cell_name in spike_times.keys(): ###########
    fire_xpos = []
    fire_ypos = []
    fire_hd = []
    for spike_time in spike_times[cell_name]:
        closest_index = np.argmin(np.abs(np.array(traj_time) - spike_time))
        # discard spikes where velocity < threshold
        if closest_index in moving_idx:
            fire_xpos.append(x_pos[closest_index])
            fire_ypos.append(y_pos[closest_index])
            fire_hd.append(head_dir[closest_index])

    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 1. Spike position diagram with color-coded head direction
    ax1.plot(x_pos, y_pos, color='lightgray', alpha=0.7, linewidth=1)  # Trajectory in light gray

    cmap = plt.get_cmap('rainbow')
    norm = Normalize(vmin=min(fire_hd), vmax=max(fire_hd))
    scatter = ax1.scatter(fire_xpos, fire_ypos, c=fire_hd, cmap=cmap, norm=norm, s=20, zorder=2)

    ax1.set_title(f'Rat Trajectory and Spike Positions for {cell_name}')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.axis('equal')  # To ensure the aspect ratio is 1:1

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Head Direction')

    # 2. Rate map
    num_bins = 50

    # Create 2D histograms
    position_hist, x_edges, y_edges = np.histogram2d(x_pos[moving_idx], y_pos[moving_idx], bins=num_bins)
    spike_hist, _, _ = np.histogram2d(fire_xpos, fire_ypos, bins=[x_edges, y_edges])

    # Calculate rate map (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        firing_map = np.divide(spike_hist, position_hist)
        firing_map[np.isinf(firing_map)] = 0
        firing_map[np.isnan(firing_map)] = 0

    # Create 2D histogram of spike positions
    # firing_map, x_edges, y_edges = np.histogram2d(fire_xpos, fire_ypos, bins=num_bins)
    sigma = 1.3  # Adjust this value to control the amount of smoothing
    firing_map_smooth = gaussian_filter(firing_map, sigma)
    firing_map_smooth = np.ma.array(firing_map_smooth.T)

    im = ax2.imshow(firing_map_smooth, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                    cmap='viridis', interpolation='bilinear') # norm=LogNorm(vmin=1)

    ax2.plot(x_pos, y_pos, color='gray', alpha=0.5, linewidth=0.5)
    # ax2.scatter(fire_xpos, fire_ypos, color='red', s=10, alpha=0.5)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Number of Spikes')

    ax2.set_title(f'Firing Map and Trajectory for {cell_name}')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')

    ax2.set_aspect('equal', adjustable='box')

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(f'Figures/PV_inactivation/mouse-{mouse_num}/{inactivation}/{cell_name}_{inactivation}.png')