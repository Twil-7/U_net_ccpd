import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2


def create_network():

    inputs = keras.layers.Input((192, 192, 3))

    # First extract feature map  1/2
    conv1 = keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='same')(inputs)
    lk1 = keras.layers.LeakyReLU()(conv1)
    conv2 = keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='same')(lk1)
    pool1 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv2)
    lk2 = keras.layers.LeakyReLU()(pool1)
    bn1 = keras.layers.BatchNormalization()(lk2)

    # Second extract feature map  1/4
    conv3 = keras.layers.Conv2D(64, kernel_size=5, strides=1, padding='same')(bn1)
    lk3 = keras.layers.LeakyReLU()(conv3)
    conv4 = keras.layers.Conv2D(64, kernel_size=5, strides=1, padding='same')(lk3)
    pool2 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv4)
    lk4 = keras.layers.LeakyReLU()(pool2)
    bn2 = keras.layers.BatchNormalization()(lk4)

    # Third extract feature map  1/8
    conv5 = keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same')(bn2)
    lk5 = keras.layers.LeakyReLU()(conv5)
    conv6 = keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same')(lk5)
    pool3 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv6)
    lk6 = keras.layers.LeakyReLU()(pool3)
    bn3 = keras.layers.BatchNormalization()(lk6)

    # Fourth extract feature map  1/16
    conv7 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(bn3)
    lk7 = keras.layers.LeakyReLU()(conv7)
    conv8 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(lk7)
    pool4 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv8)
    lk8 = keras.layers.LeakyReLU()(pool4)
    bn4 = keras.layers.BatchNormalization()(lk8)

    # Fifth extract feature map  1/32
    conv9 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(bn4)
    lk9 = keras.layers.LeakyReLU()(conv9)
    conv10 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(lk9)
    pool5 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(conv10)
    lk10 = keras.layers.LeakyReLU()(pool5)
    bn5 = keras.layers.BatchNormalization()(lk10)

    # Intermediate transition
    conv11 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(bn5)
    lk11 = keras.layers.LeakyReLU()(conv11)
    bn6 = keras.layers.BatchNormalization()(lk11)

    conv12 = keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(bn6)
    lk12 = keras.layers.LeakyReLU()(conv12)
    bn7 = keras.layers.BatchNormalization()(lk12)

    # First Deconvolution and expansion    1/16
    d_conv1 = keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(bn7)
    merge1 = keras.layers.concatenate([bn4, d_conv1])
    lk13 = keras.layers.LeakyReLU()(merge1)
    bn8 = keras.layers.BatchNormalization()(lk13)

    # Second Deconvolution and expansion    1/8
    d_conv2 = keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(bn8)
    merge2 = keras.layers.concatenate([bn3, d_conv2])
    lk14 = keras.layers.LeakyReLU()(merge2)
    bn9 = keras.layers.BatchNormalization()(lk14)

    # Third Deconvolution and expansion    1/4
    d_conv3 = keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(bn9)
    merge3 = keras.layers.concatenate([bn2, d_conv3])
    lk15 = keras.layers.LeakyReLU()(merge3)
    bn10 = keras.layers.BatchNormalization()(lk15)

    # Fourth Deconvolution and expansion    1/2
    d_conv4 = keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(bn10)
    merge4 = keras.layers.concatenate([bn1, d_conv4])
    lk16 = keras.layers.LeakyReLU()(merge4)
    bn11 = keras.layers.BatchNormalization()(lk16)

    # Fifth Deconvolution and expansion    1/1
    d_conv4 = keras.layers.Conv2DTranspose(11, kernel_size=5, strides=2, padding='same')(bn11)
    lk17 = keras.layers.LeakyReLU()(d_conv4)
    bn12 = keras.layers.BatchNormalization()(lk17)

    # Final process and Crop
    d_conv5 = keras.layers.Conv2DTranspose(2, kernel_size=5, strides=1, padding='same')(bn12)
    outputs = keras.layers.Activation('softmax')(d_conv5)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


