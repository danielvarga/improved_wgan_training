import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.layernorm

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

def Discriminator_factory(disc_type, DIM, INPUT_SHAPE, DO_BATCHNORM=False, OUTPUT_COUNT=1, WEIGHT_NOISE_SIGMA=None):

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

    if disc_type == "conv":
        return Discriminator
    elif disc_type == "resnet":
        return ResnetDiscriminator



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


