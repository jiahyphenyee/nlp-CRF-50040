from scipy.optimize import fmin_l_bfgs_b
import numpy as np

from part3 import compute_crf_loss, compute_gradients
from  part1 import get_emis_trans_dict
from helper import load_data, create_str, get_tags, get_evaluate_pos

def callbackF(w):
    '''
    This function will only be called by "fmin_l_bfgs_b"
    Arg:
        w: weights, numpy array
    '''
    loss =  compute_crf_loss(partial_train_data, tags, f)
    print('Loss:{0:.4f}'.format(loss))


def get_loss_grad(w, *args):
    '''
    This function will only be called by "fmin_l_bfgs_b"
    Arg:
        w: weights, numpy array
    Returns:
        loss: loss, float
        gradients: gradient, np array
    '''
    data, f, tags = args
    gradients = np.zeros(len(f))

    for ind, key in enumerate(f.keys()):
        f[key] = w[ind]

    loss = compute_crf_loss(data, tags, f, 0.1)
    f_gradient = compute_gradients(data, tags, f, 0.1)

    for ind, key in enumerate(f.keys()):
        gradients[ind] = f_gradient[key]

    return loss, gradients

if __name__ == '__main__':
    partial_path = '/data/partial/'
    partial_train_data = load_data(partial_path + 'train')

    tags = get_tags(partial_path + 'train', 1)
    f = get_emis_trans_dict(partial_train_data)

