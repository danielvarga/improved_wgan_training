import os, sys
sys.path.append(os.getcwd())

import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.plot

import losses


MODE = 'wgan-gp' # wgan or wgan-gp or wgan-gs
DATASET = 'gaussians' # gaussians, swissroll
GAUSSIAN_COUNT = 16
DIM = 512 # Model dimensionality
LAMBDA = .1 # Smaller lambda makes things faster for toy tasks, but isn't
            # necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 256 # Batch size
ITERS = 100000 # how many generator iterations to train for
DIRNAME="pictures"
if not os.path.exists(DIRNAME):
    os.mkdir(DIRNAME)

lib.print_model_settings(locals().copy())

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    output = tf.nn.relu(output)
    return output

def Generator(n_samples):
    # if FIXED_GENERATOR:
    #    return real_data + (1.*tf.random_normal(tf.shape(real_data)))
    noise = tf.random_normal([n_samples, 2])
    output = ReLULayer('Generator.1', 2, DIM, noise)
    output = ReLULayer('Generator.2', DIM, DIM, output)
    output = ReLULayer('Generator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Generator.4', DIM, 2, output)
    return output

def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 2, DIM, inputs)
    output = ReLULayer('Discriminator.2', DIM, DIM, output)
    output = ReLULayer('Discriminator.3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Discriminator.4', DIM, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[None, 2])
fake_data = Generator(BATCH_SIZE)



disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

# WGAN loss
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

if MODE in ('wgan-gp', 'wgan-gs'):
    alpha_strategy = "uniform"

    gen_cost, disc_cost, initial_slopes, final_slopes = losses.calculate_losses(
            BATCH_SIZE, real_data,
            Generator, Discriminator,
            MODE, alpha_strategy, LAMBDA)

disc_params = lib.params_with_name('Discriminator')
gen_params = lib.params_with_name('Generator')

if MODE == 'wgan-gp':
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(
        disc_cost, 
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(
            gen_cost, 
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()

else:
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
        disc_cost, 
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
            gen_cost, 
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()


    # Build an op to do the weight clipping
    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

print "Generator params:"
for var in lib.params_with_name('Generator'):
    print "\t{}\t{}".format(var.name, var.get_shape())
print "Discriminator params:"
for var in lib.params_with_name('Discriminator'):
    print "\t{}\t{}".format(var.name, var.get_shape())

def grid_points(count):
    scale = 2.
    size = int(np.sqrt(count))
    centers = []
    for x in xrange(-(size//2), size-size//2):
        for y in xrange(-(size//2), size-size//2):
            centers.append([x,y])
    return scale * np.array(centers)

def gaussian_centers_8():
    scale = 2.
    centers = [
            (1,0),
            (-1,0),
            (0,1),
            (0,-1),
            (1./np.sqrt(2), 1./np.sqrt(2)),
            (1./np.sqrt(2), -1./np.sqrt(2)),
            (-1./np.sqrt(2), 1./np.sqrt(2)),
            (-1./np.sqrt(2), -1./np.sqrt(2))
        ]
    centers = [(scale*x,scale*y) for x,y in centers]
    return np.array(centers)


def evaluate_mixture(iteration):
    batch_count = 10
    samples = []
    for i in range(batch_count):
        mb_samples, = session.run(
            [fake_data]
        )
        samples += mb_samples.tolist()

    samples = np.array(samples)

    assert DATASET == 'gaussians'
    if GAUSSIAN_COUNT == 8:
        centers = gaussian_centers_8()
    else:
        centers = grid_points(GAUSSIAN_COUNT)

    deltas = samples[:, np.newaxis, :] - centers[np.newaxis, :, :]

    d = np.linalg.norm(deltas, axis=2)
    clusters = d.argmin(axis=1)
    distances = d.min(axis=1)
    center_for_each = centers[clusters, :]
    offsets = samples - center_for_each

    plt.clf()
    plt.scatter(offsets[:, 0],    offsets[:, 1],    c='green', marker='+')
    plt.savefig(DIRNAME+'/offsets'+str(iteration)+'.jpg')


    unique, counts = np.unique(clusters, return_counts=True)
    # assert unique.tolist() == range(8)
    probs = counts.astype(np.float32) / len(clusters)
    print "entropy", -np.sum(probs * np.log2(probs)),

    print np.histogram(offsets[:, 0])

    import scipy.stats
    print "center", np.mean(offsets, axis=0),
    offsets -= np.mean(offsets, axis=0)
    print "std", np.std(offsets, axis=0)
    offsets /= np.std(offsets, axis=0)

    print "Kolmogorov-Smirnov of normalized coord0", tuple(scipy.stats.kstest(offsets[:, 0], 'norm'))
    print "Kolmogorov-Smirnov of normalized coord1", tuple(scipy.stats.kstest(offsets[:, 1], 'norm'))


def generate_image(true_dist, iteration):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
    points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
    points = points.reshape((-1,2))
    samples, disc_map = session.run(
        [fake_data, disc_real],
        feed_dict={real_data:points}
    )
    disc_map = session.run(disc_real, feed_dict={real_data:points})

    plt.clf()

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(x,y,disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',  marker='+')
    plt.scatter(samples[:, 0],    samples[:, 1],    c='green', marker='+')

    plt.savefig(DIRNAME+'/frame_'+str(iteration)+'.jpg')
    evaluate_mixture(iteration)

# Dataset iterator
def inf_train_gen():
    # if DATASET == '25gaussians':
    
    #     dataset = []
    #     for i in xrange(100000/25):
    #         for x in xrange(-2, 3):
    #             for y in xrange(-2, 3):
    #                 point = np.random.randn(2)*0.05
    #                 point[0] += 2*x
    #                 point[1] += 2*y
    #                 dataset.append(point)
    #     dataset = np.array(dataset, dtype='float32')
    #     dataset /= np.std(dataset)
    #     while True:
    #         np.random.shuffle(dataset)
    #         for i in xrange(len(dataset)/BATCH_SIZE):
    #             yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    if DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE, 
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5 # stdev plus a little
            yield data

    elif DATASET == 'gaussians':
        if GAUSSIAN_COUNT == 8:
            centers = gaussian_centers_8()
        else:
            centers = grid_points(GAUSSIAN_COUNT)

        while True:
            dataset = []
            for i in xrange(BATCH_SIZE):
                point = np.random.randn(2) * 0.02 # !!!!
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= np.std(dataset)
            yield dataset

# Train loop!
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    gen = inf_train_gen()
    for iteration in xrange(ITERS):
        # Train generator
        if iteration > 0:
            _data = gen.next()
            _ = session.run(gen_train_op, feed_dict={real_data: _data})
        # Train critic
        for i in xrange(CRITIC_ITERS):
            _data = gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            if MODE == 'wgan':
                _ = session.run([clip_disc_weights])
        # Write logs and save samples
        lib.plot.plot('disc cost', _disc_cost)
        if iteration % 100 == 99:
            lib.plot.flush()
            generate_image(_data, iteration+1)
        lib.plot.tick()
