'''

visualization for Allen institute rat-hippocampus achilles dataset.

'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.collections import LineCollection

import cebra.datasets
import cebra
from cebra import CEBRA

hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')

spikes = hippocampus_pos.neural
pos = hippocampus_pos.continuous_index[:, 0]


# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the 1D position as gray lines
ax.plot(range(len(pos)), pos, color='gray', alpha=0.5)

# Plot spikes for each neuron
# for i in range(spikes.shape[1]):
i = 0
spike_times = np.where(spikes[:, i] > 0)[0]
spike_positions = pos[spike_times]
ax.scatter(spike_times, spike_positions, color='red', s=1, alpha=0.5)

# Set labels and title
ax.set_xlabel('Time')
ax.set_ylabel('1D Position')
ax.set_title('1D Position with Spike Overlay')

# Show the plot
plt.tight_layout()
plt.show()