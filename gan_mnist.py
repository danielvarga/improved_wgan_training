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

# from tensorflow.contrib.tensorboard.plugins import projector

import util
import losses
import gan_logging

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp or wgan-gs
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
WEIGHT_DECAY = 0.0
ITERS = 20000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
DO_BATCHNORM = True if (MODE=='wgan') else False
ACTIVATION_PENALTY = 0.0
USE_DENSE_DISCRIMINATOR = False
OPTIMIZE_SLOPE=False
AGGREGATOR = tf.reduce_max
aggregator_names = {
    tf.reduce_max: "max",
    tf.reduce_mean: "mean"
}
SAVE_GENERATED=False

if DO_BATCHNORM:
    assert MODE=='wgan', "please don't use batchnorm for modes other than wgan, we don't know what would happen"
DIRNAME="pictures"
if not os.path.exists(DIRNAME):
    os.mkdir(DIRNAME)

# this placeholder controls whether we add some noise to the weights of convolutional filters in the discriminator
WEIGHT_NOISE_SIGMA = tf.placeholder_with_default(tf.constant(0.0), shape=[])

# Set lambda to zero at this iteration; set for -1 to disable
LAMBDA_TO_ZERO_ITER = -1 
if len(sys.argv) == 2:
    LAMBDA_TO_ZERO_ITER = int(sys.argv[1])

lib.print_model_settings(locals().copy())

def PReLU(_x, name):
    with tf.variable_scope("prelu_alphas") as scope:
        try:
            alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        except ValueError:
            scope.reuse_variables()
            alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output, fused=False)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output, fused=False)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output, fused=False)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, remember_last_activation=False):
    global last_layer_activation
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output, fused=False)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output, fused=False)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    if remember_last_activation:
        last_layer_activation = output
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

def Dense_Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 28*28])

    output = lib.ops.linear.Linear('Discriminator.1', 28*28, 1000, output)
    output = LeakyReLU(output)
    output = lib.ops.linear.Linear('Discriminator.2', 1000, 1000, output)
    output = LeakyReLU(output)
    output = lib.ops.linear.Linear('Discriminator.output', 1000, 1, output)
    return tf.reshape(output, [-1])

lambda_tf = tf.placeholder(tf.float32, shape=[])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

if USE_DENSE_DISCRIMINATOR:
    Discriminator = Dense_Discriminator

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

disc_filters = [param for param_name, param in lib._params.iteritems() if param_name.startswith("Discriminator") and (param_name.endswith("Filters") or param_name.endswith("W"))]
disc_names = [param_name for param_name, param in lib._params.iteritems() if param_name.startswith("Discriminator") and (param_name.endswith("Filters") or param_name.endswith("W"))]
disc_output_weights = [param for param_name, param in lib._params.iteritems() if param_name.startswith("Discriminator") and "Output.W" in param_name][0]

def activation_to_loss(activation):
    return tf.reduce_mean(tf.maximum(0.0,tf.square(activation) - 1.0))

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_optimizer = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    )
    gen_gvs = gen_optimizer.compute_gradients(gen_cost, var_list=gen_params)
    gen_train_op = gen_optimizer.apply_gradients(gen_gvs)

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

elif MODE in ('wgan-gp', 'wgan-gs'):
    alpha_strategy = "uniform"

    gen_cost, disc_cost, initial_slopes, final_slopes, gradient_penalty = losses.calculate_losses(
        BATCH_SIZE, real_data,
        Generator, Discriminator,
        MODE, alpha_strategy, LAMBDA, AGGREGATOR,
        WEIGHT_DECAY=WEIGHT_DECAY, params_for_wd=disc_params,
        remember_last_activation=True
    )

    gen_optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    )
    gen_gvs = gen_optimizer.compute_gradients(gen_cost, var_list=gen_params)
    gen_train_op = gen_optimizer.apply_gradients(gen_gvs)

    disc_optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    )
    disc_gvs = disc_optimizer.compute_gradients(disc_cost, var_list=disc_params)
    disc_train_op = disc_optimizer.apply_gradients(disc_gvs)

    # monitor gradient normalization effect on individual weights
    #    slope_grad_by_weight = tf.gradients(gradient_penalty, disc_filters)
    slope_grad_by_weight = tf.gradients(initial_slopes, disc_filters)
    slope_jacobian_by_weight = util.jacobian(initial_slopes, disc_output_weights)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, 
        labels=tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_optimizer = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    )
    gen_gvs = gen_optimizer.compute_gradients(gen_cost, var_list=gen_params)
    gen_train_op = gen_optimizer.apply_gradients(gen_gvs)

    disc_optimizer = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    )
    disc_gvs = disc_optimizer.compute_gradients(disc_cost, var_list=disc_params)
    disc_train_op = disc_optimizer.apply_gradients(disc_gvs)

    clip_disc_weights = None
else:
    assert False, "unknown MODE"

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)), 
        '{}/samples_{}.png'.format(DIRNAME, frame)
    )
    

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

# Train loop
with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    if MODE=='wgan':
        session_name = "%s-batchnorm=%s" % (MODE, DO_BATCHNORM)
    elif MODE=='dcgan':
        session_name = "%s" % (MODE)
    else:
        session_name = "%s-%s-lambda%.2f-%s" % (MODE, alpha_strategy, LAMBDA, aggregator_names[AGGREGATOR])
    LOG_DIR = "logs/%s" % session_name

    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    gan_logging.log_weights_grads(None, None, lib._params)

    tf.summary.scalar("disc_cost", disc_cost)

    ALPHA_COUNT = 100
    alphas, real_data_ph, slopes_for_alphas = gan_logging.log_slopes(BATCH_SIZE, OUTPUT_DIM, ALPHA_COUNT, Generator, Discriminator, fixed_noise_samples)

    # plot discriminator accuracy
    gan_logging.log_disc_accuracy(disc_real, disc_fake, BATCH_SIZE)

    merged_summary_op = tf.summary.merge_all()

    gen = inf_train_gen()

    for heldout_minibatch, _ in dev_gen():
        break

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration == LAMBDA_TO_ZERO_ITER:
            LAMBDA = 0
            print ('Lambda set to zero at iteration %d' % iteration)
        #LAMBDA = 1.0 / (iteration + 1)

        BLAMBDA = LAMBDA #np.array([LAMBDA]*1)
        #print BLAMBDA.shape

        if iteration > 0:
            _ = session.run(gen_train_op,
                            feed_dict={
                                real_data:_data,
                                WEIGHT_NOISE_SIGMA:0.0}
            )

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data, lambda_tf: BLAMBDA, WEIGHT_NOISE_SIGMA:0.0}
            )

            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        alpha_grid = np.tile(np.linspace(0, 1, ALPHA_COUNT), (BATCH_SIZE, 1))

        # Calculate dev loss and generate samples every 100 iters
        if iteration < 5 or iteration % 100 == 0:
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost, 
                    feed_dict={real_data: images, lambda_tf: BLAMBDA, WEIGHT_NOISE_SIGMA:0.0}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)
            
            # # get slopes
            # _slopes_for_alphas = session.run([slopes_for_alphas],
            #                                 feed_dict={alphas:alpha_grid,
            #                                            real_data_ph:heldout_minibatch,
            #                                            lambda_tf: BLAMBDA,
            #                                            WEIGHT_NOISE_SIGMA:0.0
            #                                 }
            # )
            # _slopes_for_alphas = _slopes_for_alphas[0]
            # _slopes_for_alphas2 = session.run([slopes_for_alphas],
            #                                 feed_dict={alphas:alpha_grid,
            #                                            real_data_ph:heldout_minibatch,
            #                                            lambda_tf: BLAMBDA,
            #                                            WEIGHT_NOISE_SIGMA:0.01
            #                                 }
            # )
            # _slopes_for_alphas2 = _slopes_for_alphas2[0]
            # print("Slopes before noise: ", np.mean(_slopes_for_alphas[:,::20], axis=0))
            # print("Slopes after  noise: ", np.mean(_slopes_for_alphas2[:,::20], axis=0))

            
            # alpha_to_disc_cost = session.run([alpha_to_disc_cost_op],
            #     feed_dict={
            #                 alphas: alpha_grid,
            #                 real_data_ph: heldout_minibatch,
            #                 lambda_tf: BLAMBDA})
            # alpha_to_disc_cost = alpha_to_disc_cost[0].reshape((BATCH_SIZE, ALPHA_COUNT))
            #            print alpha_to_disc_cost.shape, alpha_to_disc_cost[:11]
            
            #            np.save(os.path.join(LOG_DIR, 'alpha_to_disc_cost_gp05_%d.npy' % iteration), alpha_to_disc_cost)
            #            print '----- Alpha_to_disc_cost numpy array saved -----'

            if (iteration % 500 == 0) and (iteration > 0):
                saver = tf.train.Saver()
                saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), iteration)

                if MODE in ('wgan-gp', 'wgan-gs'):
                    _disc_filters, _slope_grad_by_weight = session.run([disc_filters, slope_grad_by_weight],
                                                                       feed_dict={real_data:images}
                                                                   )
                    for i in range(len(_disc_filters)):
                        filter = _disc_filters[i]
                        grads = _slope_grad_by_weight[i]
                        name = disc_names[i].replace("/", ".")
                        filename = DIRNAME + "/grad_by_weight_{}_{}.png".format(name,iteration)
                        util.scatterWithMarginals(filter.flatten(), grads.flatten(), name, filename)

                        # plt.clf()
                        # plt.scatter(filter.flatten(), grads.flatten(), c='green', marker='+')
                        # plt.savefig(filename)
 

            if SAVE_GENERATED and (iteration > 0) and (iteration % 10000  == 0):
                # save 200 * BATCH_SIZE generated images
                fakes = []
                for i in range(200):
                    fake=(session.run(fake_data))
                    fakes.append(fake.copy())
                fakes = np.concatenate(fakes, axis=0)
                filename = "/mnt/g2big/generated_images/mnist/generated_{}_{}.npy".format(MODE, iteration)
                print "Saving generated samples to {}".format(filename)
                np.save(filename, fakes)

            summary = session.run([merged_summary_op],
                feed_dict={ real_data: heldout_minibatch,
                            real_data_ph: heldout_minibatch,
                            alphas: alpha_grid,
                            lambda_tf: BLAMBDA,
                            WEIGHT_NOISE_SIGMA:0.0
                            })
            summary_writer.add_summary(summary[0], iteration)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 0):
            lib.plot.flush()

        lib.plot.tick()

    if MODE in ('wgan-gp', 'wgan-gs'): # save last layer activations, weights and activations
        images, _ = dev_gen().next()
        _disc_output_weights, _slope_jacobian, _last_layer_activation, _slope = session.run([disc_output_weights, slope_jacobian_by_weight, last_layer_activation, initial_slopes],
                                                                                            feed_dict={real_data:images}
        )
        _disc_output_weights = np.squeeze(_disc_output_weights)
        _slope_jacobian = np.squeeze(_slope_jacobian)
        np.savez(DIRNAME+"/last_layer.npz", activations=_last_layer_activation, weights=_disc_output_weights, slopes=_slope, jacobian=_slope_jacobian)
            


    if OPTIMIZE_SLOPE:
        fake_images = session.run(fixed_noise_samples)
        for real_images, _ in dev_gen():
            break

        fake_images, fake_slopes, fake_output = util.find_greatest_slopes(Discriminator, fake_images, 0, 1e-3, session)
        real_images, real_slopes, real_output = util.find_greatest_slopes(Discriminator, real_images, 0, 1e-3, session)
        for iter in range(100):
            print "real iter {}, avg slope {}, max slope: {}, avg output {}".format(iter, np.mean(real_slopes), np.max(real_slopes), np.mean(real_output))
            print "fake iter {}, avg slope {}, max slope: {}, avg output {}".format(iter, np.mean(fake_slopes), np.max(fake_slopes), np.mean(fake_output))
            lib.save_images.save_images(
                real_images.reshape((-1, 28, 28)), 
                '{}/slope_climbers_real_{}.png'.format(DIRNAME, iter)
            )
            lib.save_images.save_images(
                fake_images.reshape((-1, 28, 28)), 
                '{}/slope_climbers_fake_{}.png'.format(DIRNAME, iter)
            )
            fake_images, fake_slopes, fake_output = util.find_greatest_slopes(Discriminator, fake_images, 10, 1e-3, session)
            real_images, real_slopes, real_output = util.find_greatest_slopes(Discriminator, real_images, 10, 1e-3, session)
