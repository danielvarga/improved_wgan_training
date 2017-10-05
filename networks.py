from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.layernorm

from keras.datasets import cifar100
from keras.datasets import cifar10
from keras.layers import Input, Dense, Layer, Activation, Flatten, Lambda, Add
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import Callback, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K

net = 3

FUSED=False
RESNET_BLOCKS_PER_LAYER = 2

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def Normalize(name, axes, inputs):
    if ('Discriminator' in name):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=FUSED)

def Discriminator_factory(disc_type, DIM, INPUT_SHAPE, BATCH_SIZE, DO_BATCHNORM=False, OUTPUT_COUNT=1, WEIGHT_NOISE_SIGMA=None):

    CHANNEL = INPUT_SHAPE[0]
    
    def Discriminator(inputs):
        output = tf.reshape(inputs, [-1] + list(INPUT_SHAPE))

        output = lib.ops.conv2d.Conv2D('Discriminator.1',CHANNEL,DIM,5,output,stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
        output = LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
        if DO_BATCHNORM:
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output, fused=FUSED)
        output = LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
        if DO_BATCHNORM:
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output, fused=FUSED)
        output = LeakyReLU(output)

        output = tf.reshape(output, [-1, 4*4*4*DIM])
        if OUTPUT_COUNT > 1:
            output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, OUTPUT_COUNT, output)
        else:
            output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)
            output = tf.reshape(output, [-1])
        return output

    def ResnetDiscriminator(inputs):
        output = tf.reshape(inputs, [-1] + list(INPUT_SHAPE))

        output = lib.ops.conv2d.Conv2D('Discriminator.In', CHANNEL, DIM*1, 1, output, he_init=False)
        # output = lib.ops.conv2d.Conv2D('Discriminator.In', CHANNEL, DIM/2, 1, output, he_init=False)
        # for i in xrange(5):
        #     output = BottleneckResidualBlock('Discriminator.64x64_{}'.format(i), DIM/2, DIM/2, 3, output, resample=None)
        # output = BottleneckResidualBlock('Discriminator.Down1', DIM/2, DIM*1, 3, output, resample='down')
        for i in xrange(RESNET_BLOCKS_PER_LAYER):
            output = BottleneckResidualBlock('Discriminator.32x32_{}'.format(i), DIM*1, DIM*1, 3, output, resample=None)
        output = BottleneckResidualBlock('Discriminator.Down2', DIM*1, DIM*2, 3, output, resample='down')
        for i in xrange(RESNET_BLOCKS_PER_LAYER):
            output = BottleneckResidualBlock('Discriminator.16x16_{}'.format(i), DIM*2, DIM*2, 3, output, resample=None)
        output = BottleneckResidualBlock('Discriminator.Down3', DIM*2, DIM*4, 3, output, resample='down')
        for i in xrange(RESNET_BLOCKS_PER_LAYER):
            output = BottleneckResidualBlock('Discriminator.8x8_{}'.format(i), DIM*4, DIM*4, 3, output, resample=None)
        output = BottleneckResidualBlock('Discriminator.Down4', DIM*4, DIM*8, 3, output, resample='down')
        for i in xrange(RESNET_BLOCKS_PER_LAYER):
            output = BottleneckResidualBlock('Discriminator.4x4_{}'.format(i), DIM*8, DIM*8, 3, output, resample=None)

        output = tf.reshape(output, [-1, 4*4*8*DIM])

        if OUTPUT_COUNT > 1:
            output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*DIM, OUTPUT_COUNT, output)
        else:
            output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*DIM, 1, output)
            output = tf.reshape(output / 5., [-1])
        return output

    def DenseDiscriminator(inputs):
        input_dim = np.prod(INPUT_SHAPE)
        output = tf.reshape(inputs, [-1, input_dim])

        output = lib.ops.linear.Linear('Discriminator.1', input_dim, 1000, output)
        output = LeakyReLU(output)
        output = lib.ops.linear.Linear('Discriminator.2', 1000, 1000, output)
        output = LeakyReLU(output)
        output = lib.ops.linear.Linear('Discriminator.output', 1000, OUTPUT_COUNT, output)
        return output

    def CifarResnet(inputs):
        N = 3
        filter_num_config = [16, 32, 64]
        wideness = 1
        filter_num_config = [wideness * i for i in filter_num_config]
        weight_decay = 0.0 #1e-4

        def residual_drop(x, input_shape, output_shape, strides=(1, 1)):
            nb_filter = output_shape[0]
            conv = Conv2D(nb_filter, (3, 3), strides=strides,
                          padding="same", kernel_regularizer=l2(weight_decay))(x)
            conv = BatchNormalization()(conv)
            conv = Activation("relu")(conv)
            conv = Conv2D(nb_filter, (3, 3),
                          padding="same", kernel_regularizer=l2(weight_decay))(conv)
            conv = BatchNormalization()(conv)
            
            if strides[0] >= 2:
                x = AveragePooling2D(strides)(x)
            if (output_shape[0] - input_shape[0]) > 0:
                pad_shape = (1,
                             output_shape[1],
                             output_shape[2],
                             output_shape[0] - input_shape[0])
                padding = K.zeros(pad_shape)
                padding = K.repeat_elements(padding, BATCH_SIZE, axis=0)
                x = Lambda(lambda y: K.concatenate([y, padding], axis=3),
                           output_shape=(output_shape[1], output_shape[2], output_shape[0]))(x)

            out = Add()([conv, x])
            out = Activation("relu")(out)

            gate = K.variable(1.0, dtype="float32")
            return Lambda(lambda tensors: K.switch(gate, tensors[0], tensors[1]),
                          output_shape=(output_shape[1], output_shape[2], output_shape[0]))([out, x])
            

        def build_net(inputs, filter_num_config, nb_classes=10):
            global net
            print(inputs.shape)
            net = Conv2D(filter_num_config[0], (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(inputs)
            net = BatchNormalization()(net)
            net = Activation("relu")(net)

            for i in range(N):
                net = residual_drop(net, input_shape=(filter_num_config[0], 32, 32), output_shape=(filter_num_config[0], 32, 32))

            net = residual_drop(
                net,
                input_shape=(filter_num_config[0], 32, 32),
                output_shape=(filter_num_config[1], 16, 16),
                strides=(2, 2)
            )
            for i in range(N - 1):
                net = residual_drop(
                    net,
                    input_shape=(filter_num_config[1], 16, 16),
                    output_shape=(filter_num_config[1], 16, 16)
                )

            net = residual_drop(
                net,
                input_shape=(filter_num_config[1], 16, 16),
                output_shape=(filter_num_config[2], 8, 8),
                strides=(2, 2)
            )
            for i in range(N - 1):
                net = residual_drop(
                    net,
                    input_shape=(filter_num_config[2], 8, 8),
                    output_shape=(filter_num_config[2], 8, 8)
                )

            pool = AveragePooling2D((8, 8))(net)
            flatten = Flatten()(pool)

            predictions = Dense(nb_classes, activation=None, kernel_regularizer=l2(weight_decay))(flatten)
            predictions = Activation("softmax")(predictions)

            return predictions

        output = tf.reshape(inputs, [-1] + list(INPUT_SHAPE))
        output = build_net(output, filter_num_config=filter_num_config, nb_classes=10)
        return output



    if disc_type == "conv":
        return Discriminator
    elif disc_type == "resnet":
        return ResnetDiscriminator
    elif disc_type == "dense":
        return DenseDiscriminator
    elif disc_type == "cifarResnet":
        return CifarResnet



def BottleneckResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim/2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim/2, output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim/2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2,  output_dim=output_dim/2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim/2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)


