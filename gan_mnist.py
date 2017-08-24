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

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
DO_BATCHNORM = False
if DO_BATCHNORM:
    assert MODE=='wgan', "please don't use batchnorm for modes other than wgan, we don't know what would happen"

lib.print_model_settings(locals().copy())

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
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if DO_BATCHNORM:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

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

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    lower_alpha, upper_alpha = 0.0, 1.0
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=lower_alpha,
        maxval=upper_alpha
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

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

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    if not os.path.exists('pictures'):
        os.mkdir('pictures')
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)), 
        'pictures/samples_{}.png'.format(frame)
    )
    

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    if MODE=='wgan-gp':
        session_name = "%s-%f-%f-lambda%f" % (MODE, lower_alpha, upper_alpha, LAMBDA)
    elif MODE=='wgan':
        session_name = "%s-batchnorm=%s" % (MODE, DO_BATCHNORM)

    summary_writer = tf.summary.FileWriter("logs/%s" % session_name, graph=tf.get_default_graph())

    for param_name, param in lib._params.iteritems():
        print param_name, param
        tf.summary.histogram(param_name+"/weights", param)

    tf.summary.scalar("disc_cost", disc_cost)

    ALPHA_COUNT = 100

    alphas = tf.placeholder(tf.float32, shape=(BATCH_SIZE, ALPHA_COUNT))
    alphas1 = tf.expand_dims(alphas, axis=-1)
    real_data_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 784))
    real_data_ph1 = tf.expand_dims(real_data_ph, axis=1)
    fake_data = Generator(BATCH_SIZE)
    fake_data = tf.expand_dims(fake_data, axis=1)

    x = alphas1*fake_data + (1-alphas1)*real_data_ph1

    alpha_to_disc_cost_op = Discriminator(x)

    grad_by_alphas = tf.gradients(alpha_to_disc_cost_op, alphas)[0]

    grad_by_x = tf.gradients(Discriminator(x), [x])[0]
    slopes_for_alphas = tf.sqrt(tf.reduce_sum(tf.square(grad_by_x), reduction_indices=[1]))

    tf.summary.histogram("slopes_for_all_alphas", slopes_for_alphas)
    tf.summary.histogram("slopes_for_alpha0", slopes_for_alphas[:, 0])
    tf.summary.histogram("slopes_for_alpha1", slopes_for_alphas[:, -1])

    tf.summary.histogram("unidirectional_grad_at_all_alphas", grad_by_alphas)
    tf.summary.histogram("unidirectional_grad_at_alpha0", grad_by_alphas[:, 0])
    tf.summary.histogram("unidirectional_grad_at_alpha1", grad_by_alphas[:, -1])

    tf.summary.image("generated", tf.reshape(fixed_noise_samples, (128, 28, 28, 1)), max_outputs=50)
    

    merged_summary_op = tf.summary.merge_all()

    gen = inf_train_gen()

    for heldout_minibatch, _ in dev_gen():
        break

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run(gen_train_op)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        alpha_grid = np.tile(np.linspace(0, 1, ALPHA_COUNT), (BATCH_SIZE, 1))

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost, 
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

            
            alpha_to_disc_cost = session.run([alpha_to_disc_cost_op],
                feed_dict={
                            alphas: alpha_grid,
                            real_data_ph: heldout_minibatch})
            alpha_to_disc_cost = alpha_to_disc_cost[0].reshape((BATCH_SIZE, ALPHA_COUNT))
            print alpha_to_disc_cost.shape, alpha_to_disc_cost[:11]
            if iteration % 2000 == 1999:
                np.save('alpha_to_disc_cost_%d.npy' % iteration, alpha_to_disc_cost)
                print '----- Alpha_to_disc_cost numpy array saved -----'
            print "==="

            summary = session.run([merged_summary_op],
                feed_dict={ real_data: heldout_minibatch,
                            real_data_ph: heldout_minibatch,
                            alphas: alpha_grid
                            })
            summary_writer.add_summary(summary[0], iteration)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
