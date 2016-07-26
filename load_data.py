from scipy.io import loadmat
import numpy as np
import re
from os import listdir
from os.path import join

def load_data(path):
    mask_rawdata = loadmat('ChannelType.mat')
    gradiom_mask = np.array(map(lambda x: x[0][0] == 'MEG GRAD',mask_rawdata['Type']))
    return np.concatenate([extract_grad_mat(join(path,f),gradiom_mask) for f in listdir(path) if f.endswith(".mat")],axis=0)


def extract_grad_mat(path,gradiom_mask):
    data=loadmat(path)
    return np.transpose(data['data'][gradiom_mask])[np.newaxis,...] #additional dimension for easier concatenation to 3d array in the future

if __name__== '__main__':
    data = load_data('../meg_data/em_06_SI')
    print('It\'s fun!')