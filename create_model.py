# /**
#  * @author Zhirui(Alex) Yang
#  * @email zy2494@columbia.edu
#  * @create date 2022-12-27 23:38:36
#  * @modify date 2022-12-27 23:38:36
#  * @desc [Create three models: base_model, one_zoom_model, three_zooms_model]
#  */


import random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.utils import plot_model

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def create_base_model(patch_len=299, summary=True):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', 
                            input_shape=(patch_len, patch_len, 3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    if summary:
        model.summary()
    return model


def create_one_zoom_model(patch_len=299, trainable=False, summary=True):
    inception_zoom1 = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(patch_len, patch_len, 3))

    # freeze the inception model to increase training speed
    inception_zoom1.trainable = trainable

    model_zoom1 = models.Sequential()
    model_zoom1.add(inception_zoom1)
    model_zoom1.add(layers.GlobalAveragePooling2D())

    input_zoom1 = layers.Input(shape=(patch_len, patch_len, 3))

    encoded_zoom1 = model_zoom1(input_zoom1)

    dense1 = layers.Dense(128, activation='relu')(encoded_zoom1)
    drop_layer = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(32, activation='relu')(drop_layer)

    output = layers.Dense(1, activation='sigmoid')(dense2)
    model = models.Model(inputs=[input_zoom1], outputs=output)

    if summary:
        model.summary()
    
    return model


def create_three_zooms_model(patch_len=299, trainable=False, summary=True):
    inception_zoom1 = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(patch_len, patch_len, 3))

    inception_zoom2 = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(patch_len, patch_len, 3))

    inception_zoom3 = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(patch_len, patch_len, 3))

    # freeze the inception model to increase training speed
    inception_zoom1.trainable = trainable
    inception_zoom2.trainable = trainable
    inception_zoom3.trainable = trainable

    # creat the base model
    model_zoom1 = models.Sequential()
    model_zoom1.add(inception_zoom1)
    model_zoom1.add(layers.GlobalAveragePooling2D())

    model_zoom2 = models.Sequential()
    model_zoom2.add(inception_zoom2)
    model_zoom2.add(layers.GlobalAveragePooling2D())

    model_zoom3 = models.Sequential()
    model_zoom3.add(inception_zoom3)
    model_zoom3.add(layers.GlobalAveragePooling2D())

    input_zoom1 = layers.Input(shape=(patch_len, patch_len, 3), name="input1")
    input_zoom2 = layers.Input(shape=(patch_len, patch_len, 3), name="input2")
    input_zoom3 = layers.Input(shape=(patch_len, patch_len, 3), name="input3")

    encoded_zoom1 = model_zoom1(input_zoom1)
    encoded_zoom2 = model_zoom2(input_zoom2)
    encoded_zoom3 = model_zoom3(input_zoom3)

    merged = layers.concatenate([encoded_zoom1, encoded_zoom2, encoded_zoom3])
    dense1 = layers.Dense(256, activation='relu')(merged)
    drop_layer = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(126, activation='relu')(drop_layer)

    output = layers.Dense(1, activation='sigmoid')(dense2)
    model = models.Model(inputs=[input_zoom1, input_zoom2, input_zoom3], outputs=output)

    if summary:
        model.summary()
        
    return model