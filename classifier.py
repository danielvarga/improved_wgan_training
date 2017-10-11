import time
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
import gan_logging


LAMBDA = 0 # 1e-4 # Gradient penalty lambda hyperparameter
WEIGHT_DECAY = 0
GRADIENT_SHRINKING = False
LIPSCHITZ_TARGET = 1.0

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
ITERS = 10000 # How many iterations to train for
DO_BATCHNORM = True
ACTIVATION_PENALTY = 0.0
ALPHA_STRATEGY = "real"
SHRINKING_REDUCTOR = "max" # "none", "max", "mean", "logsum"
COMBINE_OUTPUTS_FOR_SLOPES = True # if true we take a per-batch sampled random linear combination of the logits, and calculate the slope of that.
LEARNING_RATE_DECAY = False
if LEARNING_RATE_DECAY:
    LEARNING_RATE = 0.1
else:
    LEARNING_RATE = 1e-4
AUGMENTATION = "no"

# TARGET_DIGITS = 2, 8
# number of elements in one class, total number is twice this:
TRAIN_DATASET_SIZE = 2000
TEST_DATASET_SIZE = 10000
BALANCED = False # if true we take TRAIN_DATASET_SIZE items from each digit class
OUTPUT_COUNT = 10
DATASET="cifar10" # cifar10 / mnist
DISC_TYPE = "cifarResnet" # "conv" / "resnet" / "dense" / "cifarResnet"

def heuristic_cast(s):
    s = s.strip() # Don't let some stupid whitespace fool you.
    if s=="None":
        return None
    elif s=="True":
        return True
    elif s=="False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

for k, v in [arg.split('=', 1) for arg in sys.argv[1:]]:
    assert v != '', "Malformed command line"
    assert k.startswith('--'), "Malformed arg %s" % k
    k = k[2:]
    assert k in locals(), "Unknown arg %s" % k
    v = heuristic_cast(v)
    print "Changing argument %s from default %s to %s" % (k, locals()[k], v)
    locals()[k] = v


assert LEARNING_RATE_DECAY in (True, False)
assert COMBINE_OUTPUTS_FOR_SLOPES in (True, False)


if BALANCED:
    TOTAL_TRAIN_SIZE = TRAIN_DATASET_SIZE * 10
else:
    TOTAL_TRAIN_SIZE = TRAIN_DATASET_SIZE

SESSION_NAME = "dataset_{}-net_{}-iters_{}-train_{}-lambda_{}-wd_{}-lips_{}-combslopes_{}-lrd_{}-aug_{}-bs_{}-bn_{}-ts_{}".format(
    DATASET, DISC_TYPE, ITERS, TRAIN_DATASET_SIZE, LAMBDA, WEIGHT_DECAY, LIPSCHITZ_TARGET,
    "y" if COMBINE_OUTPUTS_FOR_SLOPES else "n", "y" if LEARNING_RATE_DECAY else "n",
    AUGMENTATION, BATCH_SIZE,
    "y" if DO_BATCHNORM else "n",
    time.strftime('%Y%m%d-%H%M%S'))

if BALANCED:
    (X_train, y_train), (X_test, y_test) = data.load_balanced(DATASET, TRAIN_DATASET_SIZE, TEST_DATASET_SIZE)    
else:
    (X_train, y_train), (X_test, y_test) = data.load_set(DATASET, TRAIN_DATASET_SIZE, TEST_DATASET_SIZE)    

INPUT_SHAPE = X_train.shape[1:]
INPUT_DIM = np.prod(INPUT_SHAPE)

#X_train = np.reshape(X_train, [-1, INPUT_DIM])
#X_test = np.reshape(X_test, [-1, INPUT_DIM])

real_gen = data.classifier_generator((X_train, y_train), BATCH_SIZE, augment=False)

lib.print_model_settings(locals().copy())


Discriminator = networks.Discriminator_factory(DISC_TYPE, DIM, INPUT_SHAPE, BATCH_SIZE, DO_BATCHNORM, OUTPUT_COUNT, REUSE_BATCHNORM=True)

real_labels = tf.placeholder(tf.uint8, shape=[BATCH_SIZE])
real_labels_onehot = tf.one_hot(real_labels, 10)

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INPUT_DIM])
disc_real = Discriminator(real_data)

if DISC_TYPE == "cifarResnet":
    disc_params = tf.trainable_variables()
else:
    disc_params = lib.params_with_name('Discriminator')



disc_filters = [param for param_name, param in lib._params.iteritems() if param_name.startswith("Discriminator") and param_name.endswith("Filters")]
# param_count = np.sum([np.prod(param.shape) for param_name, param in lib._params.iteritems()])


def activation_to_loss(activation):
    return tf.reduce_mean(tf.maximum(0.0,tf.square(activation) - 1.0))

def get_slopes(input):
    output = Discriminator(input)
    if COMBINE_OUTPUTS_FOR_SLOPES:
        output_weights = tf.random_normal((OUTPUT_COUNT,))
        gradients = tf.gradients(output * output_weights, [input])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    else:
        jacobians = util.jacobian_by_batch(output, input)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(jacobians), reduction_indices=[3]))
    return slopes

loss_list = []

assert ALPHA_STRATEGY in ("real", "random", "real_plus_noise"), "In the disciminative setup only real and random are supported"
interpolates = losses.get_slope_samples(real_data, real_data, ALPHA_STRATEGY, BATCH_SIZE)
slopes = get_slopes(interpolates)

if GRADIENT_SHRINKING:
    print "gradient shrinking"
    if SHRINKING_REDUCTOR == "mean":
        grad_norm = tf.reduce_mean(slopes)
    elif SHRINKING_REDUCTOR == "max":
        grad_norm = tf.reduce_max(slopes)
    elif SHRINKING_REDUCTOR == "logsum":
        grad_norm = tf.reduce_logsumexp(slopes)
    elif SHRINKING_REDUCTOR == "none":
        grad_norm = slopes

    disc_real /= grad_norm
    disc_real *= LIPSCHITZ_TARGET

softmax_output = tf.nn.softmax(disc_real)
disc_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=disc_real,
    labels=real_labels_onehot
))
loss_list.append(('xent_loss', disc_cost))

if LAMBDA > 0:
    #    gradient_penalty = tf.reduce_mean(tf.maximum(1.0, slopes/LIPSCHITZ_TARGET)**2)
    gradient_penalty = tf.reduce_mean(tf.maximum(0.0, slopes/LIPSCHITZ_TARGET - 1)**2)
    #    gradient_penalty = tf.reduce_mean(slopes**2)
    #    gradient_penalty = tf.reduce_mean((slopes/LIPSCHITZ_TARGET-1)**2)
    disc_cost += LAMBDA*gradient_penalty
    loss_list.append(('gradient_penalty', gradient_penalty))

if ACTIVATION_PENALTY > 0:
    activation_loss = activation_to_loss(Discriminator(interpolates / ACTIVATION_PENALTY))
    disc_cost += activation_loss
    loss_list.append(('activation_loss', activation_loss))

# weight regularization
if WEIGHT_DECAY > 0:
    with tf.variable_scope('weights_norm') as scope:
        weight_loss = tf.reduce_sum(
            input_tensor = WEIGHT_DECAY*tf.stack(
                [tf.nn.l2_loss(tf.maximum(0.01, var)) for var in disc_filters]
            ),
            name='weight_loss'
        )
    disc_cost += weight_loss
else:
    weight_loss = tf.constant(0.0)
loss_list.append(('weight_loss', weight_loss))

global_step = global_step = tf.Variable(0, trainable=False)
if LEARNING_RATE_DECAY:
    values = [LEARNING_RATE * v for v in (1.0, 0.1, 0.01)]
    boundaries = [ITERS//2, 3*ITERS//4]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
else:
    learning_rate = LEARNING_RATE


if DISC_TYPE == "cifarResnet":
    disc_optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=0.9,
        use_nesterov=True
    )
else:
    disc_optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9
    )

disc_gvs = disc_optimizer.compute_gradients(disc_cost, var_list=disc_params)
disc_train_op = disc_optimizer.apply_gradients(disc_gvs, global_step=global_step)


# Train loop
with tf.Session() as session:

    # classifier is supposed to give maximal value to its true label
    def accuracy(_disc_real, _label_real):
        return float(np.sum(np.argmax(_disc_real, axis=1) == _label_real)) / len(_label_real)

    session.run(tf.global_variables_initializer())

    LOG_DIR = "logs/%s" % SESSION_NAME
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', graph=tf.get_default_graph())

    loss_summaries = []
    extra_summaries = []

    loss_summaries.append(tf.summary.scalar("disc_cost", disc_cost))
    for (name, loss) in loss_list:
        loss_summaries.append(tf.summary.scalar(name, loss))
    merged_loss_summary_op = tf.summary.merge(loss_summaries)


    for param_name, param in lib._params.iteritems():
        print param_name, param
        extra_summaries.append(tf.summary.histogram(param_name+"/weights", param))
    for grad, var in disc_gvs:
        if grad is not None:
            extra_summaries.append(tf.summary.histogram(var.name + "/gradients", grad))
    extra_summaries.append(tf.summary.histogram("slopes", slopes))
    merged_extra_summary_op = tf.summary.merge(extra_summaries)

#    merged_summary_op = tf.summary.merge_all()

    print "NETWORK PARAMETER COUNT", np.sum([np.prod(v.shape) for v in tf.trainable_variables()])

    # # log settings in tensorboard
    # all_vars = [(k,v) for (k,v) in locals().copy().items() if (k.isupper() and k!='T' and k!='SETTINGS' and k!='ALL_SETTINGS')]
    # all_vars = sorted(all_vars, key=lambda x: x[0])
    # settings_string = "\n".join(["{}: {}".format(var_name, var_value) for var_name, var_value in all_vars])
    # print settings_string
    # settings_summary = tf.Summary(value=[
    #     tf.Summary.Value(tag="settings summary", simple_value=settings_string), 
    # ])
    # test_writer.add_summary(settings_summary)
    # train_writer.add_summary(settings_summary)




    for iteration in xrange(ITERS+1):
        start_time = time.time()

        _real_data = real_gen.next()

        _weight_loss, _disc_cost, _,  _disc_real, summary_loss = session.run(
                [weight_loss, disc_cost, disc_train_op, disc_real, merged_loss_summary_op],
                feed_dict={
                    real_data: _real_data[0], real_labels: _real_data[1]}
            )

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('train weight loss', _weight_loss)
        lib.plot.plot('time', time.time() - start_time)

        train_writer.add_summary(summary_loss, iteration)
        train_acc = accuracy(_disc_real, _real_data[1])
        train_summary_acc = tf.Summary(value=[
            tf.Summary.Value(tag="accuracy", simple_value=train_acc),
        ])
        train_writer.add_summary(train_summary_acc, iteration)

        # Calculate dev loss and generate samples every 100 iters
        if iteration <= 5 or iteration % 500 == 0:
            dev_disc_costs = []
            dev_real_disc_outputs = []
            dev_real_labels = []
            dev_real_data = []

            for _real_data_test in data.classifier_generator((X_test, y_test), BATCH_SIZE, infinity=False):
                _dev_disc_cost, _dev_real_disc_output = session.run(
                    [disc_cost, disc_real],
                    feed_dict={
                        real_data: _real_data_test[0], real_labels: _real_data_test[1]}
                )
                dev_disc_costs.append(_dev_disc_cost)
                dev_real_disc_outputs.append(_dev_real_disc_output)
                dev_real_labels.append(_real_data_test[1])
                dev_real_data.append(_real_data_test[0])

            dev_real_disc_outputs = np.concatenate(dev_real_disc_outputs)
            dev_real_labels = np.concatenate(dev_real_labels)
            dev_real_data = np.concatenate(dev_real_data)
            dev_cost = np.mean(dev_disc_costs)
            dev_acc = accuracy(dev_real_disc_outputs, dev_real_labels)
            print "TRAIN ACCURACY", train_acc
            print "DEVEL ACCURACY", dev_acc

            lib.plot.plot('dev disc cost', dev_cost)

            if iteration % 2500 == 0:
                saver = tf.train.Saver()
                saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), iteration)

            dev_summary_loss, dev_summary_extra = session.run([merged_loss_summary_op, merged_extra_summary_op],
                                                              feed_dict={
                                                                  real_data: _real_data_test[0],
                                                                  real_labels: _real_data_test[1]
                                                              })
            dev_summary_acc = tf.Summary(value=[
                tf.Summary.Value(tag="accuracy", simple_value=dev_acc), 
            ])
            test_writer.add_summary(dev_summary_loss, iteration)
            test_writer.add_summary(dev_summary_extra, iteration)
            test_writer.add_summary(dev_summary_acc, iteration)

            summary_extra = session.run(
                [merged_extra_summary_op],
                feed_dict={
                    real_data: _real_data[0], real_labels: _real_data[1]}
            )
            train_writer.add_summary(summary_extra[0], iteration)

            sys.stdout.flush()

        # Write logs every 100 iters
        if (iteration <= 5) or (iteration % 500 == 0):
            lib.plot.flush()

        lib.plot.tick()
