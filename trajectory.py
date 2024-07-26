import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def rat_trajectory(shape):
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

def rat_trajectory_mec(inactivation):
    if inactivation:
        filename = f'mec_inactivation/MEC_inactivation/Inactivation_move'
    else:
        filename = f'mec_inactivation/MEC_inactivation/Control_move'

    df = pd.read_excel(f'data/{filename}.xlsx', header=34, skiprows=[35])
    df = df.replace('-', np.nan)

    # Extract the required columns
    time = df['Recording time']
    x_center = df['X center']
    y_center = df['Y center']
    head_dir = df['Head direction']
    vel = df['Velocity']

    return time, x_center, y_center, head_dir, vel


def rat_trajectory_PV(inactivation, mousenum):
    if mousenum == 1 and not inactivation:
        filestring = 'mouse1 - base and opto/Raw data-Square50_Base1_PV5_6-Trial     1'
    elif mousenum == 1 and inactivation:
        filestring = 'mouse1 - base and opto/Raw data-Square50_Opto1_PV5_6-Trial     1'
    elif mousenum == 2 and not inactivation:
        filestring = 'mouse2 - base and opto/Raw data-Square50_Base_PV5_7-Trial     1'
    elif mousenum == 2 and inactivation:
        filestring = 'mouse2 - base and opto/Raw data-Square50_Opto_PV5_7-Trial     1'
    elif mousenum == 3 and not inactivation:
        filestring = 'mouse3 - base and chemo/Raw data-PV6_2_2_Base_DL-Trial     1'
    elif mousenum == 3 and inactivation:
        filestring = 'mouse3 - base and chemo/Raw data-PV6_2_2_Chemo_DL-Trial     1'

    filename = f'PV_inactivation/JY/{filestring}'

    df = pd.read_excel(f'data/{filename}.xlsx', header=34, skiprows=[35])
    df = df.replace('-', np.nan)

    # Extract the required columns
    time = df['Recording time']
    x_center = df['X center']
    y_center = df['Y center']
    head_dir = df['Head direction']
    vel = df['Velocity']

    return time, x_center, y_center, head_dir, vel