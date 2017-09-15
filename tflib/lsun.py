import numpy as np
import scipy.misc
import time

def make_generator(all_images, n, batch_size):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        files = range(n)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for j, i in enumerate(files):
            image = all_images[i,:,:,:]
            images[j % batch_size] = image.transpose(2,0,1)
            if j > 0 and j % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_file):

    dataset = np.load(data_file)
    n = len(dataset)
    test_size = n // 10
    return (
        make_generator(dataset[:-test_size,:,:], n - test_size, batch_size),
        make_generator(dataset[-test_size:,:,:], test_size, batch_size)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()