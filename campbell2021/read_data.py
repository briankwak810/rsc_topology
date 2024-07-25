from scipy import io
import numpy as np
import pandas as pd
from scipy import interpolate

# Load the .mat file
data = io.loadmat('AA2_190807_gain_1.mat')

spike_times = data['sp']['st'][0][0].flatten()
cluster_ids = data['sp']['cids'][0][0].flatten()
cluster_groups = data['sp']['cgs'][0][0].flatten()

# For MEC recordings (files starting with 'np')
if data['sp']['dat_path'][0][0][0].startswith('np'):
    anatomical_labels = data['anatomy']['cluster_parent'][0][0].flatten()
else:
    anatomical_labels = data['anatomy']['cluster_parent'][0][0].flatten()

# Create a DataFrame with spike information
cluster_df = pd.DataFrame({
    'cluster_id': cluster_ids,
    'anatomical_label': [anatomical_labels[i][0] for i in range(len(cluster_ids))],
    'is_good': [cluster_groups[i] == 2 for i in range(len(cluster_ids))]
})

good_clusters = cluster_df[cluster_df['is_good']]
grouped_clusters = good_clusters.groupby('anatomical_label')

spike_time_arrays = {}

# Determine the maximum spike time to set the size of the arrays
max_spike_time = np.max(spike_times)
time_bins = np.arange(0, max_spike_time + 1, 1)  # 1 ms bins

for label, group in grouped_clusters:
    label_spike_arrays = []
    for cluster_id in group['cluster_id']:
        cluster_spikes = spike_times[cluster_ids == cluster_id]
        spike_counts, _ = np.histogram(cluster_spikes, bins=time_bins)
        label_spike_arrays.append(spike_counts)
    spike_time_arrays[label] = np.array(label_spike_arrays)

# Now spike_time_arrays is a dictionary where each key is an anatomical label,
# and each value is a 2D array (time steps x clusters for that label)

# Step 3: Prepare position data
position_times = data['post'].flatten()
positions = data['posx'].flatten()

# Check if position data needs interpolation
if len(position_times) != len(time_bins) - 1:
    # Interpolate position data to match spike time bins
    f = interpolate.interp1d(position_times, positions, kind='linear', fill_value='extrapolate')
    interpolated_positions = f(time_bins[:-1])  # Use bin centers
else:
    interpolated_positions = positions

# Now you have:
# 1. spike_time_arrays: a dictionary of spike time arrays for each anatomical label
# 2. time_bins: the time bins used for the spike time arrays
# 3. interpolated_positions: position data interpolated to match the time bins

# Example of how to access the data:
for label, spikes in spike_time_arrays.items():
    print(f"Anatomical label: {label}")
    print(f"Spike array shape: {spikes.shape}")
    print(f"Number of clusters: {spikes.shape[0]}")
    print(f"Number of time steps: {spikes.shape[1]}")
    print("---")

print(f"Position data shape: {interpolated_positions.shape}")
print(f"Time bins shape: {time_bins.shape}")