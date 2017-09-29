import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot


from tensorflow.contrib.tensorboard.plugins import projector
import util
import losses
import data
import networks

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
LAMBDA = 1e-4 # Gradient penalty lambda hyperparameter
WEIGHT_DECAY_FACTOR = 0
ITERS = 100 # How many generator iterations to train for
DO_BATCHNORM = False
ACTIVATION_PENALTY = 0.0
GRADIENT_SHRINKING = False
SHRINKING_REDUCTOR = "mean" # "none", "max", "mean", "softmax"
ALPHA_STRATEGY = "real"

# TARGET_DIGITS = 2, 8
# number of elements in one class, total number is twice this:
TRAIN_DATASET_SIZE = 2000
TEST_DATASET_SIZE = 10000
BALANCED = False # if BALANCED and MULTINOMIAL then  we take TRAIN_DATASET_SIZE items from each digit class
COMBINE_OUTPUTS_FOR_SLOPES = False # if MULTINOMIAL and COMBINE_OUTPUTS_FOR_SLOPES then we take a fixed random linear combination of the softmax logits and calculate slopes on them
DATASET="mnist" # cifar10 / mnist
DISC_TYPE = "conv" # "conv" / "resnet"


if DATASET == "mnist":
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_COUNT = 10
elif DATASET == "cifar10":
    INPUT_SHAPE = (3, 32, 32)
    OUTPUT_COUNT = 10
INPUT_DIM = np.prod(INPUT_SHAPE)

if BALANCED:
    TOTAL_TRAIN_SIZE = TRAIN_DATASET_SIZE * 10
else:
    TOTAL_TRAIN_SIZE = TRAIN_DATASET_SIZE

SESSION_NAME = "classifier-iter{}-train{}-lambda{}-{}".format(ITERS, TOTAL_TRAIN_SIZE, LAMBDA, ALPHA_STRATEGY)

if BALANCED:
    (X_train, y_train), (X_test, y_test) = data.load_balanced(DATASET, TRAIN_DATASET_SIZE, TEST_DATASET_SIZE)    
else:
    (X_train, y_train), (X_test, y_test) = data.load_set(DATASET, TRAIN_DATASET_SIZE, TEST_DATASET_SIZE)    

real_gen = data.classifier_generator((X_train, y_train), BATCH_SIZE)

lib.print_model_settings(locals().copy())

Discriminator = networks.Discriminator_factory(DISC_TYPE, DIM, INPUT_SHAPE, DO_BATCHNORM, OUTPUT_COUNT)

real_labels = tf.placeholder(tf.uint8, shape=[BATCH_SIZE])
real_labels_onehot = tf.one_hot(real_labels, 10)

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INPUT_DIM])
disc_real = Discriminator(real_data)

disc_params = lib.params_with_name('Discriminator')

disc_filters = [param for param_name, param in lib._params.iteritems() if param_name.startswith("Discriminator") and param_name.endswith("Filters")]


def activation_to_loss(activation):
    return tf.reduce_mean(tf.maximum(0.0,tf.square(activation) - 1.0))

def get_slopes(input, slope_by):
    output = Discriminator(input)
    if COMBINE_OUTPUTS_FOR_SLOPES:
        output_weights = tf.random_normal((OUTPUT_COUNT,))
        gradients = tf.gradients(output * output_weights, [slope_by])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    else:
        jacobians = util.jacobian_by_batch(output, slope_by)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(jacobians), reduction_indices=[3]))
    return slopes

loss_list = []

# "if True" because I was lazy to retabulate.
if True:
    if GRADIENT_SHRINKING:
        assert False, "does not currently work, needs to be moved before disc_cost calculation"
        print "gradient shrinking"
        if SHRINKING_REDUCTOR == "mean":
            grad_norm = tf.reduce_mean(slopes)
        elif SHRINKING_REDUCTOR == "max":
            grad_norm = tf.reduce_max(slopes)
        elif SHRINKING_REDUCTOR == "softmax":
            grad_norm = tf.reduce_logsumexp(slopes)
        elif SHRINKING_REDUCTOR == "none":
            grad_norm = slopes

        disc_real /= grad_norm

    softmax_output = tf.nn.softmax(disc_real)
    disc_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=disc_real, 
            labels=real_labels_onehot
    ))
    
    loss_list.append(('xent_loss', disc_cost))

    assert ALPHA_STRATEGY in ("real", "random"), "In the disciminative setup only real and random are supported"
    interpolates = losses.get_slope_samples(real_data, real_data, ALPHA_STRATEGY, BATCH_SIZE)
    slopes = get_slopes(interpolates, interpolates)

    if LAMBDA > 0:
        # gradient_penalty = tf.reduce_mean(tf.maximum(0.0, slopes-1.)**4)
        gradient_penalty = tf.reduce_mean((slopes-1)**4)
        disc_cost += LAMBDA*gradient_penalty
        loss_list.append(('gradient_penalty', gradient_penalty))

    if ACTIVATION_PENALTY > 0:
        activation_loss = activation_to_loss(Discriminator(interpolates / ACTIVATION_PENALTY))
        disc_cost += activation_loss
        loss_list.append(('activation_loss', activation_loss))

    # weight regularization
    if WEIGHT_DECAY_FACTOR > 0:
        with tf.variable_scope('weights_norm') as scope:
            weight_loss = tf.reduce_sum(
                input_tensor = WEIGHT_DECAY_FACTOR*tf.stack(
                    [tf.nn.l2_loss(tf.maximum(0.01, var)) for var in disc_filters]
                ),
                name='weight_loss'
            )
        disc_cost += weight_loss
    else:
        weight_loss = tf.constant(0.0)
    loss_list.append(('weight_loss', weight_loss))

    disc_optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    )
    disc_gvs = disc_optimizer.compute_gradients(disc_cost, var_list=disc_params)
    disc_train_op = disc_optimizer.apply_gradients(disc_gvs)


# Train loop
with tf.Session() as session:

    # classifier is supposed to give maximal value to its true label
    def accuracy(_disc_real, _label_real):
        return float(np.sum(np.argmax(_disc_real, axis=1) == _label_real)) / len(_label_real)

    session.run(tf.global_variables_initializer())

    LOG_DIR = "logs/%s" % SESSION_NAME

    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    for param_name, param in lib._params.iteritems():
        print param_name, param
        tf.summary.histogram(param_name+"/weights", param)

    for grad, var in disc_gvs:
        if grad is not None:
            tf.summary.histogram(var.name + "/gradients", grad)

    tf.summary.scalar("disc_cost", disc_cost)

    for (name, loss) in loss_list:
        tf.summary.scalar(name, loss)

    merged_summary_op = tf.summary.merge_all()

    print "NETWORK PARAMETER COUNT", np.sum([np.prod(v.shape) for v in tf.trainable_variables()])

    for iteration in xrange(ITERS+1):
        start_time = time.time()

        _real_data = real_gen.next()

        _weight_loss, _disc_cost, _,  _disc_real = session.run(
                [weight_loss, disc_cost, disc_train_op, disc_real],
                feed_dict={real_data: _real_data[0], real_labels: _real_data[1]}
            )

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('train weight loss', _weight_loss)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration <= 5 or iteration % 500 == 0:
            dev_disc_costs = []
            dev_real_disc_outputs = []
            dev_real_labels = []

            for _real_data_test in data.classifier_generator((X_test, y_test), BATCH_SIZE, infinity=False):
                _dev_disc_cost, _dev_real_disc_output = session.run(
                    [disc_cost, disc_real],
                    feed_dict={real_data: _real_data_test[0], real_labels: _real_data_test[1]}
                )
                dev_disc_costs.append(_dev_disc_cost)
                dev_real_disc_outputs.append(_dev_real_disc_output)
                dev_real_labels.append(_real_data_test[1])

            dev_real_disc_outputs = np.concatenate(dev_real_disc_outputs)
            dev_real_labels = np.concatenate(dev_real_labels)
            print "TRAIN ACCURACY", accuracy(_disc_real, _real_data[1])
            print "DEVEL ACCURACY", accuracy(dev_real_disc_outputs, dev_real_labels)

            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            if iteration % 2500 == 0:
                saver = tf.train.Saver()
                saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), iteration)

            summary = session.run([merged_summary_op],
                                      feed_dict={
                                          real_data: _real_data_test[0],
                                          real_labels: _real_data_test[1]
                                      })

            summary_writer.add_summary(summary[0], iteration)

        # Write logs every 100 iters
        if (iteration <= 5) or (iteration % 500 == 0):
            lib.plot.flush()

        lib.plot.tick()
