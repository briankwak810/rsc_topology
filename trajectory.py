import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def rat_trajectory(shape):
    # shape = 'Hex'
    filename = f'Raw data-B6_8_1_{shape}-Trial     1'

    df = pd.read_excel(f'data/{filename}.xlsx', header=34, skiprows=[35])
    df = df.replace('-', np.nan)

    # Extract the required columns
    time = df['Recording time']
    x_center = df['X nose']
    y_center = df['Y nose']
    head_dir = df['Head direction']
    vel = df['Velocity']

    return time, x_center, y_center, head_dir, vel

'''# Create the plot
plt.figure(figsize=(10, 8))

# Create a scatter plot with points colored by time
scatter = plt.scatter(x_center, y_center, c=time, cmap='viridis', s=10)

# Add a colorbar to show the time progression
cbar = plt.colorbar(scatter)
cbar.set_label('Time (s)', rotation=270, labelpad=15)

# Add labels and title
plt.xlabel('X center (cm)')
plt.ylabel('Y center (cm)')
plt.title('(X, Y) Coordinates Evolving Across Time')

# Add arrows to show direction of movement
for i in range(len(x_center) - 1):
    plt.arrow(x_center[i], y_center[i], 
              x_center[i+1] - x_center[i], y_center[i+1] - y_center[i],
              head_width=0.3, head_length=0.3, fc='r', ec='r', alpha=0.3)

# Invert y-axis to match typical coordinate systems (optional)
plt.gca().invert_yaxis()

# Show the plot
plt.grid(True)
plt.savefig(f'trajectory_{shape}.png')
plt.show()'''
