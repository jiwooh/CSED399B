# train w4c dataset on ConvLSTM model

# refs
# https://keras.io/examples/vision/conv_lstm/

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.models import Model
from keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
# Hide tensorflow debug information
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import tensorflow as tf

import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox

from utils.data_utils import load_config
from utils.data_utils import get_cuda_memory_usage
from utils.data_utils import tensor_to_submission_file
from utils.w4c_dataloader import RainData

from visualizer import visualize_sample

# Ensure TensorFlow uses the GPU
gpus = tf.config.list_physical_devices('GPU')
print("physical devices:", len(gpus))
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)

# Load the dataset
data = RainData(
    'training',
    data_root='../weather4cast-2023-lxz/data/',
    sat_bands=['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073'],
    regions=['roxi_0004'],
    full_opera_context=1512,
    size_target_center=252,
    years=['2019'],
    splits_path='../weather4cast-2023-lxz/data/timestamps_and_splits_stage2.csv'
)
val_data = RainData(
    'validation',
    data_root='../weather4cast-2023-lxz/data/',
    sat_bands=['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073'],
    regions=['roxi_0004'],
    full_opera_context=1512,
    size_target_center=252,
    years=['2019'],
    splits_path='../weather4cast-2023-lxz/data/timestamps_and_splits_stage2.csv'
)

### Issue: Time 1, 8, 9, and sometimes 10 has different trend of data compared to other times
### May need to use only 2~7

# Define data generators for keras, using shifted frames approach
def data_generator(data, batch_size):
    while True:
        X, y = [], []
        for _ in range(batch_size):
            idx = np.random.choice(len(data), size=1)[0]
            sample = data[idx] # (3, 4, 11, 252, 252)
            # time = 2 + np.random.choice(6, size=1)[0] # only time 2~6
            X.append([sample[0][0][1], sample[0][1][1], sample[0][2][1], sample[0][3][1]])
            y.append([sample[0][0][2], sample[0][1][2], sample[0][2][2], sample[0][3][2]])
        yield np.array(X), np.array(y)

train_gen = data_generator(data, batch_size=4)
val_gen = data_generator(val_data, batch_size=4)

# xx, yy = next(val_gen)
# print("shape check:", xx.shape, yy.shape) # (4, 4, 252, 252), (4, 4, 252, 252)
# breakpoint()


## INIT ##

# Construct the input layer with no definite frame size
inp = layers.Input(shape=(4, 4, 252, 252))

# ConvLSTM2D layers
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(inp)
x = BatchNormalization()(x)
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x)
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu')(x)
x = BatchNormalization()(x)
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, activation='relu')(x)
x = BatchNormalization()(x)

x = layers.Reshape((1, 4, 252, 64))(x)

# Final Conv3D layer to output the prediction
outp = Conv3D(filters=4, kernel_size=(1, 3, 3), activation='sigmoid', padding='same')(x)

# Build the complete model
model = keras.models.Model(inp, outp)
model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(),
)


## TRAIN ##

# Define callbacks to improve training
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters
epochs = 1 # 20
batch_size = 4 # 8

# Fit the model to the training data
### ERROR
### ValueError: Input 0 of layer conv_lst_m2d is incompatible with the layer: 
###             expected ndim=5, found ndim=4. Full shape received: (None, None, None, None)
model.fit(
    train_gen,
    steps_per_epoch = len(data) // batch_size,
    epochs = epochs,
    validation_data = val_gen,
    validation_steps = len(val_data) // batch_size,
    callbacks = [early_stopping, reduce_lr]
)


## PREDICTION RESULT ##

# Select a random example from the validation dataset
idx = np.random.choice(range(val_data.__len__()), size=1)[0]
example = val_data.__getitem__(idx)

# Visualize sample
visualize_sample(example)

# Predict and visualize result
prediction = model.predict(example[0][np.newaxis, ...])
visualize_sample(prediction[0])
