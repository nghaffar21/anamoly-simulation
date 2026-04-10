# This file takes the original office data(temperature and humidity) called office_sensor_data.csv and injects anomalies into the validation and test sets to simulate fire events.
# After running the program, you could open files validation_data.csv and test_data.csv to see the injected anomalies.

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import matplotlib

import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
import torch.cuda

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_fabric")

tqdm.pandas()

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# Load the dataset
features_df = pd.read_csv('../office_sensor_data.csv')

# Separate the training, validation, and test sets
def split_data():

    # Separate the training, validation, and test sets
    train_size = int(len(features_df) * 0.7)
    val_size = int(len(features_df) * 0.15)

    train_df, val_df, test_df = features_df[:train_size], features_df[train_size:train_size+val_size], features_df[train_size+val_size:]

    #print(train_df.shape, val_df.shape, test_df.shape)
    #print(features_df.shape)
    return train_df, val_df, test_df

# 1. Split the training, validation, and test sets
train_df, val_df, test_df = split_data()

def inject_anomalies(df, min_anoms=3, max_anoms=4):
    df = df.copy().reset_index(drop=True)

    n_rows = len(df)
    n_anomalies = np.random.randint(min_anoms, max_anoms + 1)

    # Create anomaly columns initialized as normal
    df['Temp_Anomaly'] = df['temperature_c'].copy()
    df['Hum_Anomaly'] = df['humidity_pct'].copy()
    df['anomaly'] = 0

    possible_indices = np.arange(n_rows - 1320) # Ensure we have enough room for the longest anomaly duration

    chosen_starts = np.random.choice(
        possible_indices,
        size=n_anomalies,
        replace=False
    )

    #anomalies_shown = 5
    for start in chosen_starts:

        # Sensor distance scenario (randomize per anomaly)
        # We can simulate different fire scenarios based on the distance of the sensor from the fire source
        distance = np.random.choice(['close', 'mid', 'far'], p=[0.3, 0.5, 0.2])

        if distance == 'close':       # 1-3m from fire
            peak_temp_increase = np.random.uniform(30, 80)
            hum_drop = np.random.uniform(10, 25)
            duration = np.random.randint(300, 900)     # 5–15 min ramp, detected fast
        elif distance == 'mid':       # 3-8m
            peak_temp_increase = np.random.uniform(10, 30)
            hum_drop = np.random.uniform(5, 12)
            duration = np.random.randint(600, 1200)
        else:                         # far / adjacent room
            peak_temp_increase = np.random.uniform(2, 10)
            hum_drop = np.random.uniform(1, 5)
            duration = np.random.randint(900, 1800)    # slow, long, subtle
        
        #duration = np.random.randint(660, 1320) # Fire duration 1-2 hours(660 - 1320 data points)
        
        print("\n", "___________ Fire Information ___________")
        print("distance: ", distance)
        print("duration: ", duration)
        print("peak_temp_increase: ", peak_temp_increase)
        print("hum_drop: ", hum_drop)

        #peak_temp_increase = np.random.uniform(8, 20) # The rise in temperature during the fire
        #hum_drop = np.random.uniform(15, 40)

        for i in range(duration):
            idx = start + i

            # Triangular smooth profile
            center = duration / 2
            scale = 1 - abs(i - center) / center

            temp_spike = peak_temp_increase * scale
            hum_spike = hum_drop * scale

            df.loc[idx, 'Temp_Anomaly'] += temp_spike
            df.loc[idx, 'Hum_Anomaly'] -= hum_spike
            #df.loc[idx, 'Hum_Anomaly'] = np.clip(df.loc[idx, 'Hum_Anomaly'], 0, 1)

            df.loc[idx, 'anomaly'] = 1
        
        #for start in chosen_starts:
        #    print("start: ", start, " | duration: ", duration, "\n")

    return df

# Inject anomalies into the validation and test sets

print("\n", "____________________ Start of Anomaly Injection for Validation set ____________________\n")
val_df = inject_anomalies(val_df)
print("\n", "____________________ End of Anomaly Injection for Validation set ____________________\n")

print("\n", "____________________ Start of Anomaly Injection for Test set ____________________\n")
test_df = inject_anomalies(test_df)
print("\n", "____________________ End of Anomaly Injection for Test set ____________________\n")

print("Validation anomaly datapoints:", val_df['anomaly'].sum())
print("Test anomaly datapoints:", test_df['anomaly'].sum())

def visualize_anomalies(df, title_suffix=""):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'Normal vs Anomaly Sensor Readings {title_suffix}', fontsize=16, fontweight='bold', y=1.01)

    time_index = np.arange(len(df))

    # ── 1. Temperature ──────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(time_index, df['temperature_c'], label='Normal Temperature', color='#01BEFE', linewidth=1, alpha=0.8)
    ax1.plot(time_index, df['Temp_Anomaly'], label='Anomaly Temperature', color='#FF006D', linewidth=1, alpha=0.8)

    # Shade anomaly regions
    anomaly_mask = df['anomaly'] == 1
    ax1.fill_between(time_index, df['temperature_c'], df['Temp_Anomaly'],
                     where=anomaly_mask, color='#FF006D', alpha=0.2, label='Anomaly Region')

    ax1.set_title('Temperature: Normal vs Anomaly', fontsize=13)
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ── 2. Humidity ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(time_index, df['humidity_pct'], label='Normal Humidity', color='#93D30C', linewidth=1, alpha=0.8)
    ax2.plot(time_index, df['Hum_Anomaly'], label='Anomaly Humidity', color='#FF7D00', linewidth=1, alpha=0.8)

    anomaly_mask = df['anomaly'] == 1
    ax2.fill_between(time_index, df['humidity_pct'], df['Hum_Anomaly'],
                     where=anomaly_mask, color='#FF7D00', alpha=0.2, label='Anomaly Region')

    ax2.set_title('Humidity: Normal vs Anomaly', fontsize=13)
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Humidity (%)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

visualize_anomalies(val_df, title_suffix="— Validation Set")
visualize_anomalies(test_df, title_suffix="— Test Set")

train_df.to_csv("train_data.csv", index=False)
val_df.to_csv("validation_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Files saved successfully.")