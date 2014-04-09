import os, numpy, overfeat

from multiprocessing import Pool
from scipy.ndimage import imread
from scipy.misc import imresize

PHOTO_PATH = 'img/google'

def read_in_photo(photo_path):
    photo = imread(photo_path)
    return to_square(photo)

def to_square(photo, dimension=231):
    '''231x231 is the size used by overfeat in sample code'''
    h0 = photo.shape[0]
    w0 = photo.shape[1]
    d0 = float(min(h0, w0))
    h1 = int(round(dimension*h0/d0))
    w1 = int(round(dimension*w0/d0))
    # TODO
    w1 = h1
    return imresize(photo, (h1, w1)).astype(numpy.float32)

def get_photo_path():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(curr_path, PHOTO_PATH)

def get_photo_files(path):
    photo_files = [os.path.join(path,f) for f in os.listdir(path)]
    return filter(os.path.isfile, photo_files)

if __name__ == '__main__':
    '''train and test scikit.svm on photo features extracted using overfeat.'''

    # read in and reshape photos
    photo_files = get_photo_files(get_photo_path())
    pool = Pool()
    photos = pool.map(read_in_photo, photo_files)

    # extract photo features by running through overfeat
    features = pool.map(overfeat.fprop, photos)

    overfeat.init('../../data/default/net_weight_0', 0)

    # train svm on photo features

    # test classification
