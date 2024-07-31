# train w4c dataset on ConvLSTM model

# refs
# https://keras.io/examples/vision/conv_lstm/

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

### TEST ###
data = RainData('validation')

print(data.__getitem__(0))