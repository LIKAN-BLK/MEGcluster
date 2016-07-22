from scipy.io import loadmat
import numpy as np
import re
from os import listdir
from os.path import join

def load_data(path):
    return np.concatenate([extract_grad_mat(join(path,f)) for f in listdir(path) if f.endswith(".mat")],axis=0)


def extract_grad_mat(path):
    data=loadmat(path)
    gradiom_mask = parse_labels(data['label'])
    return np.transpose(data['avg'][gradiom_mask])[np.newaxis,...] #additional dimension for easier concatenation to 3d array in the future


def parse_labels(array_to_parse):
    fun=lambda x: bool(re.match('MEG\d\d\d[2-3]',x[0][0]))
    return np.array(map(fun,array_to_parse))


if __name__== '__main__':
    data = load_data("E:\\em_06_SI")
    print('It\'s fun!')