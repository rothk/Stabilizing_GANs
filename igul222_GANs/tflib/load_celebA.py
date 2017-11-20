import numpy as np
import scipy.misc
from glob import glob
import fnmatch
import time
import os


def center_crop(image, input_h, input_w, resize_h, resize_w):
    h, w = image.shape[:2]
    j = int(round((h - input_h)/2.))
    i = int(round((w - input_w)/2.))
    return scipy.misc.imresize(image[j:j+input_h, i:i+input_w], [resize_h, resize_w])


def make_generator(pathnames, n_files, batch_size, crop=True):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        files = np.arange(n_files)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = scipy.misc.imread("{}".format(pathnames[i]))
            
            if crop:
                image = center_crop(image, 178, 178, 64, 64)
            else:
                image = scipy.misc.imresize(image, [64, 64])
            
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_dir, crop=True):
    pathnames_train = glob(os.path.join(data_dir, 'train', '*.jpg'))
    pathnames_val = glob(os.path.join(data_dir, 'test', '*.jpg'))

    return (
        make_generator(pathnames_train, len(pathnames_train), batch_size, crop=crop),
        make_generator(pathnames_val, len(pathnames_val), batch_size, crop=crop)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        #print("{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()
