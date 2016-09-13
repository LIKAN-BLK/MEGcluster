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
    return np.transpose(data['F'][gradiom_mask])[np.newaxis,...] #additional dimension for easier concatenation to 3d array in the future

def extract_window(data,window_start,window_end,baseline_start,baseline_end):
    #Extract window with baseline correction
    target_window = data[:,window_start:window_end:]
    baseline = np.tile(data[:,baseline_start:baseline_end:].mean(axis=1)[:,np.newaxis,:],(1,target_window.shape[1],1))
    return target_window - baseline

def get_data(path):
    #data_start_time = -1000
    #fixation_time = -180
    #baseline -1000:-500
    #window -80:320 ms

    window_start = -80-(-1000)
    window_end = window_start+400
    baseline_start = 0
    baseline_end = 300

    path_to_target = join(path, 'SI')
    path_to_nontarget = join(path, 'error')
    target_data = load_data(path_to_target) # trials x time x channel
    nontarget_data = load_data(path_to_nontarget)


    target_data = extract_window(target_data,window_start,window_end,baseline_start,baseline_end)
    nontarget_data = extract_window(nontarget_data,window_start,window_end,baseline_start,baseline_end)
    return target_data, nontarget_data
if __name__== '__main__':
    data = load_data('../meg_data/em_06_SI')
    print('It\'s fun!')