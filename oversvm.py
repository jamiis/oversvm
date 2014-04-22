import os, numpy, overfeat

from multiprocessing import Pool
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn import svm
from pprint import pprint
from collections import namedtuple
from copy import copy

PHOTO_PATH = 'img/google'

def read_in_photo(photo_path, dim=231):
    '''
    return resized photo read in from file path. 
    231x231 is the size used by overfeat in sample code.
    '''
    photo = imread(photo_path)
    photo = imresize(photo, (dim, dim))
    photo = photo.astype(numpy.float32)

    # numpy loads photo with colors as last dim, transpose tensor
    h = photo.shape[0]
    w = photo.shape[1]
    c = photo.shape[2]
    photo = photo.reshape(w*h, c)
    photo = photo.transpose()
    photo = photo.reshape(c, h, w)
    return photo

def get_photo_path():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(curr_path, PHOTO_PATH)

def get_photo_files(path):
    photo_files = [os.path.join(path,f) for f in os.listdir(path)]
    return filter(os.path.isfile, photo_files)

def process_through_net(photo, feature_layer=None):
    if not feature_layer:
        feature_layer = overfeat.get_n_layers() - 2
    likelihoods = copy(overfeat.fprop(photo))
    features    = copy(overfeat.get_output(feature_layer))
    return likelihoods, features

def top_n_predictions(likelihoods, n=1):
    '''
    returns top n predictions given an overfeat likelihoods vector whose 
    index corresponds with a category
    '''
    assert len(likelihoods) == 1000
    assert n >= 1 and n <= 1000

    Prediction = namedtuple('Prediction', ['name_index','likelihood'])
    predictions = (Prediction(i,v) for i,v in enumerate(likelihoods))

    # sort prediction by descending likelihood 
    predictions = sorted(predictions, key=lambda x: -x.likelihood)

    return [overfeat.get_class_name(p.name_index) for p in predictions[0:n]]

def print_predictions(filepaths, likelihoods, n=3):
    ''' print 'filename [predictions]' '''
    top_n = [top_n_predictions(likelys, n) for likelys in likelihoods]

    for filepath, top in zip(filepaths, top_n):
        filename = os.path.split(filepath)[1]
        predictions = "; ".join(top)
        print filename, ":\t", predictions

if __name__ == '__main__':
    '''train and test scikit.svm on photo features extracted using overfeat.'''
    # read in and reshape photos
    photo_files = get_photo_files(get_photo_path())
    photos = [read_in_photo(path) for path in photo_files]

    # extract photo features by running through overfeat
    overfeat.init("./data/default/net_weight_0", 0)
    # likelihoods, features = zip(*(process_through_net(photo) for photo in photos))
    pool = Pool()
    likelihoods, features = zip(*pool.map(process_through_net, photos))
    pool.close()
    pool.join()

    print_predictions(photo_files, likelihoods)
