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

MODE = 'wgan-gp-sigmoid' # dcgan, wgan, or wgan-gp, or wgan-gp-sigmoid
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 0.0 #1e-4 # Gradient penalty lambda hyperparameter
WEIGHT_DECAY_FACTOR = 0
ITERS = 1000 # How many generator iterations to train for
DO_BATCHNORM = False
ACTIVATION_PENALTY = 0.0
USE_DENSE_DISCRIMINIATOR = False
GRADIENT_SHRINKING = False
SHRINKING_REDUCTOR = "mean" # "none", "max", "mean", "softmax"
ALPHA_STRATEGY = "uniform"

MULTINOMIAL = True # if true, then we use all digits
TARGET_DIGITS = 2, 8
# number of elements in one class, total number is twice this:
TRAIN_DATASET_SIZE = 2000
TEST_DATASET_SIZE = 10000
BALANCED = False # if BALANCED and MULTINOMIAL then  we take TRAIN_DATASET_SIZE items from each digit class
COMBINE_OUTPUTS_FOR_SLOPES = True # if MULTINOMIAL and COMBINE_OUTPUTS_FOR_SLOPES then we take a fixed random linear combination of the softmax logits and calculate slopes on them
DATASET="cifar10" # cifar10 / mnist

if DATASET == "mnist":
    OUTPUT_SHAPE = (1, 28, 28)
elif DATASET == "cifar10":
    OUTPUT_SHAPE = (3, 28, 28)
CHANNEL = OUTPUT_SHAPE[0]
OUTPUT_DIM = np.prod(OUTPUT_SHAPE)


if not MULTINOMIAL:
    TOTAL_TRAIN_SIZE = TRAIN_DATASET_SIZE * 2
else:
    if BALANCED:
        TOTAL_TRAIN_SIZE = TRAIN_DATASET_SIZE * 10
    else:
        TOTAL_TRAIN_SIZE = TRAIN_DATASET_SIZE

SESSION_NAME = "classifier-{}-iter{}-multinomial{}-train{}-lambda{}-{}".format(MODE, ITERS, MULTINOMIAL, TOTAL_TRAIN_SIZE, LAMBDA, ALPHA_STRATEGY)

if DO_BATCHNORM:
    assert MODE=='wgan', "please don't use batchnorm for modes other than wgan, we don't know what would happen"
DIRNAME="pictures"
if not os.path.exists(DIRNAME):
    os.mkdir(DIRNAME)

# load data
if MULTINOMIAL:
    if BALANCED:
        (X_train, y_train), (X_test, y_test) = data.load_balanced(DATASET, TRAIN_DATASET_SIZE, TEST_DATASET_SIZE)    
    else:
        (X_train, y_train), (X_test, y_test) = data.load_set(DATASET, TRAIN_DATASET_SIZE, TEST_DATASET_SIZE)    
    real_gen = data.classifier_generator((X_train, y_train), BATCH_SIZE)
else:
    (reals_train, fakes_train), (reals_test, fakes_test) = data.load_pairs(
        target_digits = TARGET_DIGITS, train_dataset_size=TRAIN_DATASET_SIZE)
    real_gen = data.generator(reals_train, BATCH_SIZE)
    fake_gen = data.generator(fakes_train, BATCH_SIZE)


# this placeholder controls whether we add some noise to the weights of convolutional filters in the discriminator
WEIGHT_NOISE_SIGMA = tf.placeholder_with_default(tf.constant(0.0), shape=[])
lambda_tf = tf.placeholder_with_default(tf.constant(LAMBDA), shape=[])


# Set lambda to zero at this iteration; set for -1 to disable
LAMBDA_TO_ZERO_ITER = -1 
if len(sys.argv) == 2:
    LAMBDA_TO_ZERO_ITER = int(sys.argv[1])

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def Discriminator(inputs):
    output = tf.reshape(inputs, [-1] + list(OUTPUT_SHAPE))

    output = lib.ops.conv2d.Conv2D('Discriminator.1',CHANNEL,DIM,5,output,stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

def Classifier_Discriminator(inputs):
    output = tf.reshape(inputs, [-1] + list(OUTPUT_SHAPE))

    output = lib.ops.conv2d.Conv2D('Discriminator.1',CHANNEL,DIM,5,output,stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 10, output)
    return output


if MULTINOMIAL:
    Discriminator = Classifier_Discriminator
    real_labels = tf.placeholder(tf.uint8, shape=[BATCH_SIZE])
    real_labels2 = tf.one_hot(real_labels, 10)

fake_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
disc_fake = Discriminator(fake_data)
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
disc_real = Discriminator(real_data)



disc_params = lib.params_with_name('Discriminator')

disc_filters = [param for param_name, param in lib._params.iteritems() if param_name.startswith("Discriminator") and param_name.endswith("Filters")]


def activation_to_loss(activation):
    return tf.reduce_mean(tf.maximum(0.0,tf.square(activation) - 1.0))

def get_slopes(input, slope_by):
    output = Discriminator(input)
    if MULTINOMIAL:
        if COMBINE_OUTPUTS_FOR_SLOPES:
            output_weights = tf.random_normal((10,))
            gradients = tf.gradients(output * output_weights, [slope_by])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        else:
            jacobians = util.jacobian_by_batch(output, slope_by)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(jacobians), reduction_indices=[3]))
    else:
        gradients = tf.gradients(output, [slope_by])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    return slopes

if MODE == 'wgan':
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    disc_optimizer = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    )
    disc_gvs = disc_optimizer.compute_gradients(disc_cost, var_list=disc_params)
    disc_train_op = disc_optimizer.apply_gradients(disc_gvs)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)
    weight_loss = tf.constant(0.0)
    
elif MODE.startswith('wgan-gp'):
    if MODE == 'wgan-gp':
        if MULTINOMIAL:
            disc_cost = tf.reduce_mean(-tf.reduce_sum(real_labels2 * tf.log(disc_real), reduction_indices=[1]))
        else:
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    elif MODE == 'wgan-gp-sigmoid':
        if MULTINOMIAL:
            softmax_output = tf.nn.softmax(disc_real)
            disc_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=disc_real, 
                labels=real_labels2
            ))
        else:
            disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_fake,
                labels=tf.zeros_like(disc_fake)
            ))
            disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_real,
                labels=tf.ones_like(disc_real)
            ))
            disc_cost /= 2.

    interpolates = losses.get_slope_samples(real_data, fake_data, ALPHA_STRATEGY, BATCH_SIZE)
    slopes = get_slopes(interpolates, interpolates)
    
    if GRADIENT_SHRINKING:
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
        disc_fake /= grad_norm
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        
    if ACTIVATION_PENALTY > 0:
        activation_loss = activation_to_loss(Discriminator(interpolates / ACTIVATION_PENALTY))
        disc_cost += activation_loss
    
    if LAMBDA > 0:
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += LAMBDA*gradient_penalty

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

    HARD_MARGIN_WEIGHT = 0.0
    if HARD_MARGIN_WEIGHT != 0:
        print "!!!!!!!!!!!!!!!!"
        print "wgan-g* with hard margin loss, weight", HARD_MARGIN_WEIGHT
        hard_margin_loss = (tf.reduce_max(disc_fake) - tf.reduce_min(disc_real))
        disc_cost += HARD_MARGIN_WEIGHT * hard_margin_loss

    disc_optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    )
    disc_gvs = disc_optimizer.compute_gradients(disc_cost, var_list=disc_params)
    disc_train_op = disc_optimizer.apply_gradients(disc_gvs)

    clip_disc_weights = None

elif MODE == 'dcgan':
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real,
        labels=tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

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

    HARD_MARGIN_WEIGHT = 0.0
    if HARD_MARGIN_WEIGHT != 0:
        print "!!!!!!!!!!!!!!!!"
        print "dcgan with hard margin loss, weight", HARD_MARGIN_WEIGHT
        hard_margin_loss = tf.reduce_max(disc_fake) - tf.reduce_min(disc_real)
        disc_cost += HARD_MARGIN_WEIGHT * hard_margin_loss

    disc_optimizer = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    )
    disc_gvs = disc_optimizer.compute_gradients(disc_cost, var_list=disc_params)
    disc_train_op = disc_optimizer.apply_gradients(disc_gvs)

    clip_disc_weights = None
    weight_loss = tf.constant(0.0)


# Train loop
with tf.Session() as session:

    # discriminator is supposed to give big numbers for reals, small numbers for fakes.
    def accuracy_ranking(_disc_real, _disc_fake):
        assert len(_disc_real) == len(_disc_fake)
        n = len(_disc_real)
        flags = np.zeros(2*n)
        flags[-n:] = 1
        sorter = np.argsort(np.concatenate((_disc_fake, _disc_real)))
        rearranged = flags[sorter]
        return float(np.sum(rearranged[-n:])) / n

    # discriminator is supposed to give positive numbers for reals, negative numbers for fakes.
    def accuracy_classification(_disc_real, _disc_fake):
        return float(np.sum(_disc_real > 0) + np.sum(_disc_fake < 0)) / (len(_disc_real) + len(_disc_fake))

    # classifier is supposed to give maximal value to its true label
    def accuracy_multinomial(_disc_real, _label_real):
        return float(np.sum(np.argmax(_disc_real, axis=1) == _label_real)) / len(_label_real)

    session.run(tf.global_variables_initializer())

    LOG_DIR = "logs/%s" % SESSION_NAME

    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    # for param_name, param in lib._params.iteritems():
    #     print param_name, param
    #     tf.summary.histogram(param_name+"/weights", param)

    # for grad, var in disc_gvs:
    #     if grad is not None:
    #         tf.summary.histogram(var.name + "/gradients", grad)

    tf.summary.scalar("disc_cost", disc_cost)

    ALPHA_COUNT = 100

    alphas = tf.placeholder(tf.float32, shape=(BATCH_SIZE, ALPHA_COUNT))
    alphas1 = tf.expand_dims(alphas, axis=-1)
    real_data_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
    real_data_ph1 = tf.expand_dims(real_data_ph, axis=1)
    fake_data_ph = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
    fake_data_ph1 = tf.expand_dims(fake_data_ph, axis=1)

    # x = alphas1*fake_data_ph1 + (1-alphas1)*real_data_ph1
    # slopes_for_alphas = get_slopes(x, alphas)

    # # grad_by_alphas = tf.gradients(Discriminator(x), alphas)[0]

    # # grad_by_x = tf.gradients(Discriminator(x), [x])[0]
    # # slopes_for_alphas = tf.sqrt(tf.reduce_sum(tf.square(grad_by_x), reduction_indices=[2]))

    # x2 = tf.random_uniform(x.shape, minval=-1, maxval=1)
    # slopes_for_x2 = get_slopes(x2, x2)

    # # grad_by_x2 = tf.gradients(Discriminator(x2), [x2])[0]
    # # slopes_for_x2 = tf.sqrt(tf.reduce_sum(tf.square(grad_by_x2), reduction_indices=[2]))

    # slopes_for_x2 = tf.reshape(slopes_for_x2, [BATCH_SIZE, ALPHA_COUNT, -1])
    # slopes_for_alphas = tf.reshape(slopes_for_alphas, [BATCH_SIZE, ALPHA_COUNT, -1])

    # tf.summary.histogram("slopes_at_random", slopes_for_x2)
    
    # # tf.summary.histogram("slopes_for_all_alphas", slopes_for_alphas)
    # tf.summary.histogram("slopes_for_alpha0", slopes_for_alphas[:, 0])
    # tf.summary.histogram("slopes_for_alpha1", slopes_for_alphas[:, -1])

    merged_summary_op = tf.summary.merge_all()

    for iteration in xrange(ITERS):
        start_time = time.time()

        for i in xrange(CRITIC_ITERS):
            _real_data = real_gen.next()

            if MULTINOMIAL:
                _weight_loss, _disc_cost, _,  _disc_real = session.run(
                    [weight_loss, disc_cost, disc_train_op, disc_real],
                    feed_dict={real_data: _real_data[0], fake_data: np.random.permutation(_real_data[0]), real_labels: _real_data[1]}
                )
            else:
                _fake_data = fake_gen.next()
                _weight_loss, _disc_cost, _, _disc_real, _disc_fake = session.run(
                    [weight_loss, disc_cost, disc_train_op, disc_real, disc_fake],
                    feed_dict={real_data: _real_data, fake_data: _fake_data}
                )

            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        # if MULTINOMIAL:
        #     print "TRAIN ACCURACY MULTINOMIAL", accuracy_multinomial(_disc_real, _real_data[1])
        # else:
        #     print "TRAIN ACCURACY", accuracy(_disc_real, _disc_fake)



        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('train weight loss', _weight_loss)
        lib.plot.plot('time', time.time() - start_time)

        alpha_grid = np.tile(np.linspace(0, 1, ALPHA_COUNT), (BATCH_SIZE, 1))

        # Calculate dev loss and generate samples every 100 iters
        if iteration < 5 or iteration % 100 == 0:
            dev_disc_costs = []
            dev_real_disc_outputs = []
            dev_fake_disc_outputs = []
            dev_real_labels = []

            if MULTINOMIAL:
                for _real_data_test in data.classifier_generator((X_test, y_test), BATCH_SIZE, infinity=False):
                    _dev_disc_cost, _dev_real_disc_output = session.run(
                        [disc_cost, disc_real],
                        feed_dict={real_data: _real_data_test[0], fake_data: np.random.permutation(_real_data_test[0]), real_labels: _real_data_test[1]}
                    )
                    dev_disc_costs.append(_dev_disc_cost)
                    dev_real_disc_outputs.append(_dev_real_disc_output)
                    dev_real_labels.append(_real_data_test[1])
            else:
                for _real_data_test, _fake_data_test in zip(
                        data.generator(reals_test, BATCH_SIZE, infinity=False),
                        data.generator(fakes_test, BATCH_SIZE, infinity=False)
                ):
                    _dev_disc_cost, _dev_real_disc_output, _dev_fake_disc_output = session.run(
                        [disc_cost, disc_real, disc_fake],
                        feed_dict={real_data: _real_data_test, fake_data: _fake_data_test}
                    )
                    dev_disc_costs.append(_dev_disc_cost)
                    dev_real_disc_outputs.append(_dev_real_disc_output)
                    dev_fake_disc_outputs.append(_dev_fake_disc_output)

            dev_real_disc_outputs = np.concatenate(dev_real_disc_outputs)
            if MULTINOMIAL:
                dev_real_labels = np.concatenate(dev_real_labels)
            else:
                dev_fake_disc_outputs = np.concatenate(dev_fake_disc_outputs)

            # print "REAL", dev_real_disc_outputs[:20], dev_real_disc_outputs.shape
            # print "FAKE", dev_fake_disc_outputs[:20], dev_fake_disc_outputs.shape
            if MULTINOMIAL:
                print "DEVEL ACCURACY MULTINOMIAL", accuracy_multinomial(dev_real_disc_outputs, dev_real_labels)
            else:
                print "DEVEL ACCURACY RANKING", accuracy_ranking(dev_real_disc_outputs, dev_fake_disc_outputs)
                print "DEVEL ACCURACY CLASSIFICATION", accuracy_classification(dev_real_disc_outputs, dev_fake_disc_outputs)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

#             # get slopes
#             _slopes_for_alphas = session.run([slopes_for_alphas],
#                                             feed_dict={alphas:alpha_grid,
#                                                        real_data_ph: _real_data_test,
#                                                        fake_data_ph: _fake_data_test
#                                             }
#             )
#             _slopes_for_alphas = _slopes_for_alphas[0]
#             _slopes_for_alphas2 = session.run([slopes_for_alphas],
#                                             feed_dict={alphas:alpha_grid,
#                                                        real_data_ph: _real_data_test,
#                                                        fake_data_ph: _fake_data_test,
#                                                        WEIGHT_NOISE_SIGMA:0.01
#                                             }
#             )
#             _slopes_for_alphas2 = _slopes_for_alphas2[0]
# #            print("Slopes before noise: ", np.mean(_slopes_for_alphas[:,::20], axis=0))
# #            print("Slopes after  noise: ", np.mean(_slopes_for_alphas2[:,::20], axis=0))

            if iteration % 500 == 0:
                saver = tf.train.Saver()
                saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), iteration)

            if MULTINOMIAL:
                _fake_data = np.random.permutation(_real_data_test[0])
                summary = session.run([merged_summary_op],
                                      feed_dict={
                                          real_data: _real_data_test[0],
                                          fake_data: _fake_data,
                                          real_labels: _real_data_test[1],
                                          real_data_ph: _real_data_test[0],
                                          fake_data_ph: _fake_data,
                                          alphas: alpha_grid
                                      })
            else:
                summary = session.run([merged_summary_op],
                                      feed_dict={
                                          real_data: _real_data_test,
                                          fake_data: _fake_data_test,
                                          real_data_ph: _real_data_test,
                                          fake_data_ph: _fake_data_test,
                                          alphas: alpha_grid
                                      })
            summary_writer.add_summary(summary[0], iteration)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
