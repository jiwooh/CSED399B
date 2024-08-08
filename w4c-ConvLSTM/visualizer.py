# visualize datapoint from w4c dataset

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers

import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox

from utils.data_utils import load_config
from utils.data_utils import get_cuda_memory_usage
from utils.data_utils import tensor_to_submission_file
from utils.w4c_dataloader import RainData

import random

data = RainData(
    'training',
    data_root='../weather4cast-2023-lxz/data/',
    sat_bands=['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073'],
    regions=['boxi_0015'],
    full_opera_context=1512,
    size_target_center=252,
    years=['2019'],
    splits_path='../weather4cast-2023-lxz/data/timestamps_and_splits_stage2.csv'
)

print("dataset size:", len(data))

def visualize_sample(data, sample_idx=0, time_idx=0, num_channels=4):
    """
    Visualizes the specified number of channels for a given sample and time index.
    
    Parameters:
    data (numpy.ndarray): The data to visualize with shape (channels, time, width, height).
    sample_idx (int): The index of the sample to visualize.
    time_idx (int): The index of the time frame to visualize.
    num_channels (int): The number of channels to visualize.
    """
    sample = data[sample_idx]
    channels = sample[:num_channels, :, :, :]
    num_time_frames = sample.shape[1]

    fig, axes = plt.subplots(num_time_frames, num_channels, figsize=(15, 3 * num_time_frames))
    for t in range(num_time_frames):
        for c in range(num_channels):
            ax = axes[t, c]
            ax.imshow(channels[c, t], cmap='viridis')
            if t == 0:
                ax.set_title(f'Channel {c + 1}')
            if c == 0:
                ax.set_ylabel(f'Time {t + 1}')
            # ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set(frame_on=False)

    plt.tight_layout()
    plt.show()
    plt.savefig('d.png')

idx = random.randint(0, len(data))
d = data[idx]
visualize_sample(d)