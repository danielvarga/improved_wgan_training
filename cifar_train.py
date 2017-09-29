from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(2 ** 10)

# Prevent reaching to maximum recursion depth in `theano.tensor.grad`
import sys
sys.setrecursionlimit(2 ** 20)

from six.moves import range

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

import keras.losses as losses

import tensorflow as tf

LAMBDA = 0.001

batch_size = 128
nb_classes = 10
nb_epoch = 25*500
N = 3
weight_decay = 1e-4
lr_schedule = [0.5, 0.75]

death_mode = "lin_decay"  # or uniform
death_rate = 0.0

filter_num_config = [16, 32, 64]
wideness = 1
filter_num_config = [wideness * i for i in filter_num_config]

img_rows, img_cols = 32, 32
img_channels = 3

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
#X_test, y_test = shuffle(X_test, y_test)

X_train = X_train[:2000]
y_train = y_train[:2000]


X_train = X_train[0:batch_size * (X_train.shape[0] // batch_size)]
X_test  = X_test[0:batch_size * (X_test.shape[0] // batch_size)]
y_train = y_train[0:batch_size * (y_train.shape[0] // batch_size)]
y_test  = y_test[0:batch_size * (y_test.shape[0] // batch_size)]



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

add_tables = []



def residual_drop(x, input_shape, output_shape, strides=(1, 1)):
    global add_tables

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
        padding = K.repeat_elements(padding, batch_size, axis=0)
        print(padding.get_shape().as_list())
        x = Lambda(lambda y: K.concatenate([y, padding], axis=3),
                   output_shape=(output_shape[1], output_shape[2], output_shape[0]))(x)
    _death_rate = K.variable(death_rate)
    scale = K.ones_like(conv) - _death_rate

    conv = Lambda(lambda c: K.in_test_phase(scale * c, c),
                  output_shape=(output_shape[1], output_shape[2], output_shape[0]))(conv)

    out = Add()([conv, x])
    out = Activation("relu")(out)

    gate = K.variable(1.0, dtype="float32")
    add_tables += [{"death_rate": _death_rate, "gate": gate}]
    return Lambda(lambda tensors: K.switch(gate, tensors[0], tensors[1]),
                  output_shape=(output_shape[1], output_shape[2], output_shape[0]))([out, x])

pre = None
inp = None
loss = None
def build_net(filter_num_config, nb_classes=10):
    global inp
    global pre

    inputs = Input(shape=(img_rows, img_cols, img_channels))
    inp=inputs

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
    pre = predictions
    predictions = Activation("softmax")(predictions)

    return inputs, predictions


inputs, predictions = build_net(filter_num_config=filter_num_config, nb_classes=nb_classes)


def gp_loss(x_true, x_pred):
    loss = losses.categorical_crossentropy(x_true, x_pred)

    output_weights = tf.random_normal((nb_classes,))
    gradients = tf.gradients(output_weights * pre, [inp])[0]
    initial_slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((initial_slopes-1.0)**2)

    return loss + LAMBDA * gradient_penalty

model = Model(inputs=inputs, outputs=predictions)
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=gp_loss, metrics=["accuracy"])



def open_all_gates():
    for t in add_tables:
        K.set_value(t["gate"], 1)


# setup death rate
for i, tb in enumerate(add_tables, start=1):
    if death_mode == "uniform":
        K.set_value(tb["death_rate"], death_rate)
    elif death_mode == "lin_decay":
        K.set_value(tb["death_rate"], i / len(add_tables) * death_rate)
    else:
        raise


class GatesUpdate(Callback):
    def on_batch_begin(self, batch, logs={}):
        open_all_gates()

        rands = np.random.uniform(size=len(add_tables))
        for t, rand in zip(add_tables, rands):
            if rand < K.get_value(t["death_rate"]):
                K.set_value(t["gate"], 0)

    def on_batch_end(self, batch, logs={}):
        open_all_gates()  # for validation


def schedule(epoch_idx):
    if (epoch_idx + 1) < (nb_epoch * lr_schedule[0]):
        return 0.1
    elif (epoch_idx + 1) < (nb_epoch * lr_schedule[1]):
        return 0.01

    return 0.001

datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.,#0.125,
    height_shift_range=0.,#0.125,
    horizontal_flip=False,#True,
    vertical_flip=False)
datagen.fit(X_train)

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.,
    height_shift_range=0.,
    horizontal_flip=False,
    vertical_flip=False)
test_datagen.fit(X_test)

tbCallback = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# fit the model on the batches generated by datagen.flow()
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                    steps_per_epoch=50000//batch_size,
                    epochs=nb_epoch,
                    validation_data=test_datagen.flow(X_test, Y_test, batch_size=batch_size),
                    validation_steps=X_test.shape[0]//batch_size,
                    callbacks=[GatesUpdate(), LearningRateScheduler(schedule), tbCallback])
