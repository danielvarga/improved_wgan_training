import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.layernorm

# from keras.datasets import cifar100
# from keras.datasets import cifar10
# from keras.layers import Input, Dense, Layer, Activation, Flatten, Lambda, Add
# from keras.layers.convolutional import Conv2D, AveragePooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.models import Model
# from keras.optimizers import SGD
# from keras.regularizers import l2
# from keras.callbacks import Callback, LearningRateScheduler, TensorBoard
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import np_utils
# import keras.backend as K

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

        def residual_drop(x, input_shape, output_shape, level, block, strides=(1, 1)):

            nb_filter = output_shape[0]

            output = x

            filter_size = 3
            output = lib.ops.conv2d.Conv2D("Discriminator.Lvl{}.Block{}.Conv1".format(level, block), input_dim=input_shape[0], output_dim=nb_filter, filter_size=filter_size, inputs=output, stride=strides[0], weight_noise_sigma=WEIGHT_NOISE_SIGMA)
            if DO_BATCHNORM:
                output = lib.ops.batchnorm.Batchnorm("Discriminator.Lvl{}.Block{}.BN1".format(level, block), [0,2,3], output, fused=FUSED)
            output = LeakyReLU(output, alpha=0.0)

            output = lib.ops.conv2d.Conv2D("Discriminator.Lvl{}.Block{}.Conv2".format(level, block), nb_filter, nb_filter, filter_size, output, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
            if DO_BATCHNORM:
                output = lib.ops.batchnorm.Batchnorm("Discriminator.Lvl{}.Block{}.BN2".format(level, block), [0,2,3], output, fused=FUSED)

            if strides[0] >= 2:
                x = tf.contrib.layers.avg_pool2d(x, strides, data_format='NCHW')

            if (output_shape[0] - input_shape[0]) > 0:
                pad_shape = (BATCH_SIZE,
                             output_shape[0] - input_shape[0],
                             output_shape[1],
                             output_shape[2])
                padding = tf.zeros(shape=pad_shape)
                x = tf.concat([x, padding], axis=1)

            output = x + output
            output = LeakyReLU(output, alpha=0.0)

            return output


        def build_net(inputs, filter_num_config, nb_classes=10):
            net = lib.ops.conv2d.Conv2D("Discriminator.Lvl0.Conv0", 3, filter_num_config[0], 3, inputs, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
            if DO_BATCHNORM:
                net = lib.ops.batchnorm.Batchnorm("Discriminator.Lvl0.BN0", [0,2,3], net, fused=FUSED)
            net = LeakyReLU(net, alpha=0.0)

            for i in range(N):
                net = residual_drop(net, input_shape=(filter_num_config[0], 32, 32), output_shape=(filter_num_config[0], 32, 32), level=1, block=i)

            net = residual_drop(
                net,
                input_shape=(filter_num_config[0], 32, 32),
                output_shape=(filter_num_config[1], 16, 16),
                level=2, block=0,
                strides=(2, 2)
            )
            for i in range(1, N):
                net = residual_drop(
                    net,
                    input_shape=(filter_num_config[1], 16, 16),
                    output_shape=(filter_num_config[1], 16, 16),
                    level=2, block=i
                )

            net = residual_drop(
                net,
                input_shape=(filter_num_config[1], 16, 16),
                output_shape=(filter_num_config[2], 8, 8),
                level=3, block=0,
                strides=(2, 2)
            )
            for i in range(1,N):
                net = residual_drop(
                    net,
                    input_shape=(filter_num_config[2], 8, 8),
                    output_shape=(filter_num_config[2], 8, 8),
                    level=3, block=i
                )


            pool = tf.contrib.layers.avg_pool2d(net, 8, data_format='NCHW')
            flatten = tf.reshape(pool, [BATCH_SIZE, -1])

            predictions = lib.ops.linear.Linear('Discriminator.Linear', flatten.get_shape().as_list()[1], nb_classes, flatten)

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
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim/2, output_dim=output_dim)

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


