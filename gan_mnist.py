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

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
WEIGHT_DECAY_FACTOR = 0
ITERS = 20000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
DO_BATCHNORM = False
ACTIVATION_PENALTY = 0.0
USE_DENSE_DISCRIMINIATOR = False
GRADIENT_SHRINKING = False
SHRINKING_REDUCTOR = "mean" # "none", "max", "mean", "softmax"
lower_alpha, upper_alpha = 0.0, 1.0

if DO_BATCHNORM:
    assert MODE=='wgan', "please don't use batchnorm for modes other than wgan, we don't know what would happen"
DIRNAME="pictures"
if not os.path.exists(DIRNAME):
    os.mkdir(DIRNAME)

# this placeholder controls whether we add some noise to the weights of convolutional filters in the discriminator
WEIGHT_NOISE_SIGMA = tf.placeholder(tf.float32, shape=[])

# Set lambda to zero at this iteration; set for -1 to disable
LAMBDA_TO_ZERO_ITER = -1 
if len(sys.argv) == 2:
    LAMBDA_TO_ZERO_ITER = int(sys.argv[1])

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

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2, weight_noise_sigma=WEIGHT_NOISE_SIGMA)
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

if USE_DENSE_DISCRIMINIATOR:
    Discriminator = Dense_Discriminator

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

disc_filters = [param for param_name, param in lib._params.iteritems() if param_name.startswith("Discriminator") and param_name.endswith("Filters")]


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
    weight_loss = tf.constant(0.0)
    
elif MODE == 'wgan-gp':    
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=lower_alpha,
        maxval=upper_alpha
    )

    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    # noise = tf.random_normal([BATCH_SIZE, 128])
    # fake_images = Generator(128, noise)
    # gradients = tf.gradients(Discriminator(fake_images), [noise])[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    
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
        gen_cost = -tf.reduce_mean(disc_fake)            
        
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
    weight_loss = tf.constant(0.0)

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
    #session.run(tf.initialize_all_variables())

    if MODE=='wgan-gp':
        if GRADIENT_SHRINKING:
            session_name = "%s-%.2f-%.2f-lambda%.2f-%s" % (MODE, lower_alpha, upper_alpha, LAMBDA, SHRINKING_REDUCTOR)
        else:
            session_name = "%s-%.2f-%.2f-lambda%.2f" % (MODE, lower_alpha, upper_alpha, LAMBDA)
    elif MODE=='wgan':
        session_name = "%s-batchnorm=%s" % (MODE, DO_BATCHNORM)
    elif MODE=='dcgan':
        session_name = "%s" % (MODE)

    LOG_DIR = "logs/%s" % session_name

    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

    for param_name, param in lib._params.iteritems():
        print param_name, param
        tf.summary.histogram(param_name+"/weights", param)

    for grad, var in gen_gvs:
        if grad is not None:
            tf.summary.histogram(var.name + "/gradients", grad)
    for grad, var in disc_gvs:
        if grad is not None:
            tf.summary.histogram(var.name + "/gradients", grad)

    """
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = fixed_noise_samples.name
    embedding.metadata_path = os.path.join(LOG_DIR, fixed_noise_samples.name)
    projector.visualize_embeddings(summary_writer, config)
    """

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
    slopes_for_alphas = tf.sqrt(tf.reduce_sum(tf.square(grad_by_x), reduction_indices=[2]))

    x2 = tf.random_uniform(x.shape, minval=-1, maxval=1)
    grad_by_x2 = tf.gradients(Discriminator(x2), [x2])[0]
    slopes_for_x2 = tf.sqrt(tf.reduce_sum(tf.square(grad_by_x2), reduction_indices=[2]))
    tf.summary.histogram("slopes_at_random", slopes_for_x2)
    
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
            _weight_loss, _disc_cost, _ = session.run(
                [weight_loss, disc_cost, disc_train_op],
                feed_dict={real_data: _data, lambda_tf: BLAMBDA, WEIGHT_NOISE_SIGMA:0.0}
            )

            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('train weight loss', _weight_loss)
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
            
            # get slopes
            _slopes_for_alphas = session.run([slopes_for_alphas],
                                            feed_dict={alphas:alpha_grid,
                                                       real_data_ph:heldout_minibatch,
                                                       lambda_tf: BLAMBDA,
                                                       WEIGHT_NOISE_SIGMA:0.0
                                            }
            )
            _slopes_for_alphas = _slopes_for_alphas[0]
            _slopes_for_alphas2 = session.run([slopes_for_alphas],
                                            feed_dict={alphas:alpha_grid,
                                                       real_data_ph:heldout_minibatch,
                                                       lambda_tf: BLAMBDA,
                                                       WEIGHT_NOISE_SIGMA:0.01
                                            }
            )
            _slopes_for_alphas2 = _slopes_for_alphas2[0]
            print("Slopes before noise: ", np.mean(_slopes_for_alphas[:,::20], axis=0))
            print("Slopes after  noise: ", np.mean(_slopes_for_alphas2[:,::20], axis=0))

            
            # alpha_to_disc_cost = session.run([alpha_to_disc_cost_op],
            #     feed_dict={
            #                 alphas: alpha_grid,
            #                 real_data_ph: heldout_minibatch,
            #                 lambda_tf: BLAMBDA})
            # alpha_to_disc_cost = alpha_to_disc_cost[0].reshape((BATCH_SIZE, ALPHA_COUNT))
#            print alpha_to_disc_cost.shape, alpha_to_disc_cost[:11]

#            np.save(os.path.join(LOG_DIR, 'alpha_to_disc_cost_gp05_%d.npy' % iteration), alpha_to_disc_cost)
#            print '----- Alpha_to_disc_cost numpy array saved -----'

            if iteration % 500 == 0:
                saver = tf.train.Saver()
                saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), iteration)

            summary = session.run([merged_summary_op],
                feed_dict={ real_data: heldout_minibatch,
                            real_data_ph: heldout_minibatch,
                            alphas: alpha_grid,
                            lambda_tf: BLAMBDA,
                            WEIGHT_NOISE_SIGMA:0.0
                            })
            summary_writer.add_summary(summary[0], iteration)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
