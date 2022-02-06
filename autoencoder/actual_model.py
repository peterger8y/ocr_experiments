import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import keras
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D, Conv2DTranspose
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform
from resnet_building_blocks import identity_block, convolutional_block, convolutional_block_deconv


def ResNet50_deconv(input_shape=(100, 300, 3)):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block_deconv(X, f=3, filters=[512, 128, 128], stage=1, block='a', s=2)

    X = convolutional_block_deconv(X, f=3, filters=[256, 64, 64], stage=2, block='a', s=2)

    X = Conv2DTranspose(1, (7, 7), strides=(2, 2), name='deconv3', kernel_initializer=glorot_uniform(seed=0))(X)

    X = Activation('sigmoid')(X)


    # X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

if __name__ == '__main__':
    model = ResNet50_deconv()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()