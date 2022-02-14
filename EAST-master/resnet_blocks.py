import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, \
    Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, MaxPool2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform


def unpool(inputs):
    return tf.image.resize(
        inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2], method='bilinear',
        preserve_aspect_ratio=False, antialias=False, name=None
    )


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])  # SKIP Connection
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def EAST(input_shape=(512, 512, 3), batch=1, style='RBOX'):
    text_scale = 512
    X_input = Input(shape=input_shape, batch_size=batch)

    X = ZeroPadding2D((3, 3))(X_input)
    # changed initial convolution from (7, 7) to (6, 6) in order to facilitate recombination later on...
    # feel free to experiment to find cleaner, programatic solution
    X = Conv2D(64, (6, 6), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X_1 = MaxPooling2D((3, 3), strides=(2, 2))(X)
    print(X_1)

    X = convolutional_block(X_1, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X_2 = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    X = convolutional_block(X_2, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X_3 = identity_block(X, 3, [128, 128, 512], stage=3, block='d')


    X = convolutional_block(X_3, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X_4 = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X_4, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X_5 = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    f = [X_5, X_4, X_3, X_2]
    for i in range(4):
        print('Shape of f_{} {}'.format(i, f[i].shape))
    g = [None, None, None, None]
    h = [None, None, None, None]
    num_outputs = [None, 128, 64, 32]
    for i in range(4):
        if i == 0:
            h[i] = f[i]
        else:
            pre = tf.concat([g[i - 1], f[i]], axis=-1)
            c1_1 = Conv2D(num_outputs[i], (1, 1), strides=(1, 1), padding='same',
                          kernel_initializer=glorot_uniform(seed=0))(pre)
            h[i] = Conv2D(num_outputs[i], (3, 3), strides=(1, 1),
                          kernel_initializer=glorot_uniform(seed=0), padding='same')(c1_1)
        if i <= 2:
            g[i] = unpool(h[i])
        else:
            g[i] = Conv2D(num_outputs[i], (3, 3), strides=(1, 1), padding='same',
                          kernel_initializer=glorot_uniform(seed=0))(h[i])
        print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

    F_score = Conv2D(1, (1, 1), strides=(1, 1), activation='sigmoid')(g[3])
    # 4 channel of axis aligned bbox and 1 channel rotation angle
    geo_map = Conv2D(4, (1, 1), strides=(1, 1), padding='same', activation='sigmoid')(g[3]) * text_scale
    angle_map = (Conv2D(1, (1, 1), strides=(1, 1), padding='same', activation='sigmoid')(
        g[3]) - 0.5) * np.pi / 2  # angle is between [-45, 45]
    F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    model = Model(inputs=X_input, outputs=(F_score, F_geometry))
    return model


if __name__ == '__main__':
    model = EAST()
    print(model.output_shape)
