import numpy as np
from keras.datasets import mnist, cifar10

def load_raw_data(dataset):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = np.expand_dims(X_train, 3)
        X_test = np.expand_dims(X_test, 3)

    elif dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

    # convert brightness values from bytes to floats between 0 and 1:
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_train = featurewise_std_normalization(featurewise_center(X_train))
    X_test = featurewise_std_normalization(featurewise_center(X_test))

    return (X_train, y_train), (X_test, y_test)


def load_pairs(dataset, target_digits, train_dataset_size):
    (X_train, y_train), (X_test, y_test) = load_raw_data(dataset)

    reals_train = X_train[y_train==target_digits[0]]
    fakes_train = X_train[y_train==target_digits[1]]

    # circa 5500 is the untruncated size. 200 seems to be the sweet spot for wgan-gp.
    reals_train = reals_train[:train_dataset_size]
    fakes_train = fakes_train[:train_dataset_size]

    reals_test  = X_test [y_test ==target_digits[0]]
    fakes_test  = X_test [y_test ==target_digits[1]]

    print "TARGET DIGITS", target_digits
    print "TRAIN DATA SIZE", len(reals_train), len(fakes_train)
    print "TEST DATA SIZE", len(reals_test), len(fakes_test)

    return (reals_train, fakes_train), (reals_test, fakes_test)

def load_set(dataset, TRAIN_DATASET_SIZE, TEST_DATASET_SIZE):
    (X_train, y_train), (X_test, y_test) = load_raw_data(dataset)
    X_train = X_train[:TRAIN_DATASET_SIZE]
    y_train = y_train[:TRAIN_DATASET_SIZE]
    X_test = X_test[:TEST_DATASET_SIZE]
    y_test = y_test[:TEST_DATASET_SIZE]
    return (X_train, y_train), (X_test, y_test)


def load_balanced(dataset, size_per_digit, TEST_DATASET_SIZE):
    (X_train, y_train), (X_test, y_test) = load_raw_data(dataset)

    X_train_selected = []
    y_train_selected = []
    for digit in range(10):
        xs = X_train[y_train==digit]
        ys = y_train[y_train==digit]
        X_train_selected.append(xs[:size_per_digit])
        y_train_selected.append(ys[:size_per_digit])
    X_train = np.concatenate(X_train_selected)
    y_train = np.concatenate(y_train_selected)

    X_test = X_test[:TEST_DATASET_SIZE]
    y_test = y_test[:TEST_DATASET_SIZE]

    return (X_train, y_train), (X_test, y_test)

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

def classifier_generator((xs, ys), batch_size, infinity=True):
    while True:
        perm = np.random.permutation(len(xs))
        xs2 = xs[perm]
        ys2 = ys[perm]
        i = 0
        while (i+1) * batch_size <= len(xs2):
            yield (xs2[i * batch_size : (i+1) * batch_size], ys2[i * batch_size : (i+1) * batch_size])
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
