import random
import numpy as np
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def inf_gen(dataset, batch_size, limit=None):
    if dataset == "25gaussians":
        return lambda: gen_25gaussians(batch_size, limit)
    elif dataset == "swissroll":
        return lambda: gen_swissroll(batch_size, limit)
    elif dataset == "8gaussians":
        return lambda: gen_8gaussians(batch_size, limit)
    else:
        raise ValueError, "No such toy dataset defined: {0}" % dataset

# Dataset iterators
def gen_25gaussians(batch_size, limit=None):
    dataset = []
    for i in xrange(100000/25):
        for x in xrange(-2, 3):
            for y in xrange(-2, 3):
                point = np.random.randn(2)*0.05
                point[0] += 2*x
                point[1] += 2*y
                dataset.append(point)
    dataset = np.array(dataset, dtype='float32')
    np.random.shuffle(dataset)
    dataset /= 2.828 # stdev
    while True:
        for i in xrange(len(dataset)/batch_size):
            yield dataset[i*batch_size:(i+1)*batch_size], None

def gen_swissroll(batch_size, limit=None):
    while True:
        data = sklearn.datasets.make_swiss_roll(
            n_samples=batch_size, 
            noise=0.25
        )[0]
        data = data.astype('float32')[:, [0, 2]]
        data /= 7.5 # stdev plus a little
        yield data, None

def gen_8gaussians(batch_size, limit=None):
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
    iter = 0
    while limit is None or iter * batch_size <= limit:
        dataset = []
        for i in xrange(batch_size):
            point = np.random.randn(2)*.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414 # stdev
        iter += 1
        yield dataset, None

def generate_image(session, true_dist, real_data, fake_data, disc_real, frame_index):
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

    plt.savefig("frame-{%03d}.jpg" % frame_index)
