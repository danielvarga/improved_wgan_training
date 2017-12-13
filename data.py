import numpy as np
from keras.datasets import mnist, cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file
import gzip
import os



def load_raw_data(dataset, seed=None):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = np.expand_dims(X_train, 3)
        X_test = np.expand_dims(X_test, 3)

    elif dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
        X_train = np.expand_dims(X_train, 3)
        X_test = np.expand_dims(X_test, 3)

    elif dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
    else:
        assert False, "Unknown dataset: " + dataset

    # convert brightness values from bytes to floats between 0 and 1:
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print "Using CHANNELS_FIRST convention!!!"
    X_train = np.transpose(X_train, axes=(0,3,1,2))
    X_test = np.transpose(X_test, axes=(0,3,1,2))

    # save last 10000 from X_train, y_train for development set
    if dataset == "cifar10" or dataset == "cifar100":
        X_devel = X_test
        y_devel = y_test
    else:
        X_devel = X_train[-10000:]
        y_devel = y_train[-10000:]
        X_train = X_train[:-10000]
        y_train = y_train[:-10000]

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
        sh = np.random.permutation(len(X_train))
        X_train = X_train[sh]
        y_train = y_train[sh]
        sh = np.random.permutation(len(X_test))
        X_test = X_test[sh]
        y_test = y_test[sh]
        np.random.set_state(state)

    # since we have already been using the test set for calibration, we switch the roles of the devel set and the test set
    X_devel2 = X_test
    y_devel2 = y_test
    X_test2 = X_devel
    y_test2 = y_devel

    return (X_train, y_train), (X_devel2, y_devel2), (X_test2, y_test2)


# def load_pairs(dataset, target_digits, train_dataset_size):
#     (X_train, y_train), (X_test, y_test) = load_raw_data(dataset)

#     reals_train = X_train[y_train==target_digits[0]]
#     fakes_train = X_train[y_train==target_digits[1]]

#     # circa 5500 is the untruncated size. 200 seems to be the sweet spot for wgan-gp.
#     reals_train = reals_train[:train_dataset_size]
#     fakes_train = fakes_train[:train_dataset_size]

#     reals_test  = X_test [y_test ==target_digits[0]]
#     fakes_test  = X_test [y_test ==target_digits[1]]

#     print "TARGET DIGITS", target_digits
#     print "TRAIN DATA SIZE", len(reals_train), len(fakes_train)
#     print "TEST DATA SIZE", len(reals_test), len(fakes_test)

#     return (reals_train, fakes_train), (reals_test, fakes_test)

def load_set(dataset, TRAIN_DATASET_SIZE, DEVEL_DATASET_SIZE, TEST_DATASET_SIZE, seed=None):
    if dataset == "toy2":
        return load_toy(TRAIN_DATASET_SIZE, DEVEL_DATASET_SIZE, TEST_DATASET_SIZE, 2, seed=seed)
    elif dataset == "toy1":
        return load_toy(TRAIN_DATASET_SIZE, DEVEL_DATASET_SIZE, TEST_DATASET_SIZE, 1, seed=seed)

    (X_train, y_train), (X_devel, y_devel), (X_test, y_test) = load_raw_data(dataset, seed=seed)
    assert len(X_train) >= TRAIN_DATASET_SIZE
    assert len(X_test) >= TEST_DATASET_SIZE
    assert len(X_devel) >= DEVEL_DATASET_SIZE
    X_train = X_train[:TRAIN_DATASET_SIZE]
    y_train = y_train[:TRAIN_DATASET_SIZE]
    X_devel = X_devel[:DEVEL_DATASET_SIZE]
    y_devel = y_devel[:DEVEL_DATASET_SIZE]
    X_test = X_test[:TEST_DATASET_SIZE]
    y_test = y_test[:TEST_DATASET_SIZE]
    return (X_train, y_train), (X_devel, y_devel), (X_test, y_test)


# def load_balanced(dataset, size_per_digit, TEST_DATASET_SIZE, seed=None):
#     (X_train, y_train), (X_test, y_test) = load_raw_data(dataset)

#     X_train_selected = []
#     y_train_selected = []
#     for digit in range(10):
#         xs = X_train[y_train==digit]
#         ys = y_train[y_train==digit]
#         X_train_selected.append(xs[:size_per_digit])
#         y_train_selected.append(ys[:size_per_digit])
#     X_train = np.concatenate(X_train_selected)
#     y_train = np.concatenate(y_train_selected)

#     X_test = X_test[:TEST_DATASET_SIZE]
#     y_test = y_test[:TEST_DATASET_SIZE]

#     return (X_train, y_train), (X_test, y_test)

def generator(data, batch_size, infinity=True):
    d = data.copy()
    while True:
        np.random.shuffle(d)
        i = 0
        while (i+1) * batch_size <= len(d):
            yield d[i * batch_size : (i+1) * batch_size]
            i += 1
        if not infinity:
            break

def regression_generator((xs, ys), batch_size, infinity=True):
    indices = np.array(range(len(xs)))
    while True:
        np.random.shuffle(indices)
        i = 0
        while (i+1) * batch_size <= len(indices):
            x = xs[i * batch_size : (i+1) * batch_size]
            y = ys[i * batch_size : (i+1) * batch_size]
            yield x, y
            i += 1
        if not infinity:
            break


def classifier_generator((xs, ys), batch_size, infinity=True, augment=False):
    if augment:
        datagen = ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=False,
            featurewise_std_normalization=True,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.125,
            height_shift_range=0.125,
            horizontal_flip=True,
            vertical_flip=False,
            data_format="channels_first")
        datagen.fit(xs)
    else:
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
            vertical_flip=False,
            data_format="channels_first")
        datagen.fit(xs)

    while True:
        i = 0
        gen = datagen.flow(xs, ys, batch_size=batch_size, shuffle=True)
        while (i+1) * batch_size <= len(xs):
            x, y = gen.next()
            yield np.reshape(x, [batch_size, -1]), y
            i += 1
        if not infinity:
            break

def featurewise_center(xs):
    assert xs.ndim == 4
    mean = np.mean(xs, axis=(0, 1, 2))
    mean = np.expand_dims(mean, 0)
    mean = np.expand_dims(mean, 0)
    mean = np.expand_dims(mean, 0)
    return xs - mean

def featurewise_std_normalization(xs):
    assert xs.ndim == 4
    std = np.std(xs, axis=(0, 1, 2))
    std = np.expand_dims(std, 0)
    std = np.expand_dims(std, 0)
    std = np.expand_dims(std, 0)
    return xs / (std + 1e-7)




def load_fashion_mnist():
    """Loads the Fashion-MNIST dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'fashion-mnist')
    base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for file in files:
        paths.append(get_file(file, origin=base + file, cache_subdir=dirname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


def load_toy(TRAIN_DATASET_SIZE, DEVEL_DATASET_SIZE, TEST_DATASET_SIZE, dim, seed=None):
    noise_sigma=0 #0.1
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    X_train = np.random.uniform(low=-1, size=(TRAIN_DATASET_SIZE, dim))

    if dim == 2:
        # dev set is a (-1,1) grid
        w = int(np.sqrt(DEVEL_DATASET_SIZE))
        X_devel = np.zeros(shape=(w * w, 2))
        for x1 in range(w):
            for x2 in range(w):
                X_devel[w * x1 + x2] = float(x1) / w, float(x2) / w
        X_devel = 2 * X_devel - 1

        # dev set is a small grid around one training point 
        # TODO to be removed
#        X_devel = X_train[5] + 0.001 * X_devel
    
        # y = x1^2 + x2^2
        def f(xs):
            xs_squared = np.square(xs)
            ys = np.sum(np.square(xs), axis=1)
            return np.reshape(ys, [len(ys), 1])
    elif dim == 1:
        X_devel = np.arange(-1, 1, 2.0 / DEVEL_DATASET_SIZE)
        X_devel = np.reshape(X_devel, [DEVEL_DATASET_SIZE, 1])
        def f(xs):
            ys = np.square(xs)
            return np.reshape(ys, [len(ys), 1])
    
    y_train = f(X_train)
    y_devel = f(X_devel)
    X_test = X_devel
    y_test = y_devel

    y_train += noise_sigma * np.random.normal(size=y_train.shape)


    return (X_train, y_train), (X_devel, y_devel), (X_test, y_test)

                           

