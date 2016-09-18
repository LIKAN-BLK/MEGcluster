from scipy.io import loadmat
import numpy as np
import re
from os import listdir
from os.path import join


def get_data(path,sensors_type):
    # sensor_type -  'MEG GRAD' or 'MEG MAG'
    path_to_target = join(path, 'SI')
    path_to_nontarget = join(path, 'error')
    target_data = load_data(path_to_target,sensors_type)
    nontarget_data = load_data(path_to_nontarget,sensors_type)
    return target_data, nontarget_data

def load_data(path,sensor_type):
    # sensor_type -  'MEG GRAD' or 'MEG MAG'
    mask_rawdata = loadmat('ChannelType.mat')
    gradiom_mask = np.array(map(lambda x: x[0][0] == sensor_type,mask_rawdata['Type']))
    return np.concatenate([extract_grad_mat(join(path,f),gradiom_mask) for f in listdir(path) if f.endswith(".mat")],axis=0)


def extract_grad_mat(path,gradiom_mask):
    data=loadmat(path)
    return np.transpose(data['F'][gradiom_mask])[np.newaxis,...] #additional dimension for easier concatenation to 3d array in the future

def extract_window(data,window_start,window_end,baseline_start,baseline_end):
    #Extract window with baseline correction
    target_window = data[:,window_start:window_end:]
    baseline = np.tile(data[:,baseline_start:baseline_end:].mean(axis=1)[:,np.newaxis,:],(1,target_window.shape[1],1))
    return target_window - baseline

def get_data(path,sensor_type):
    # sensor_type -  'MEG GRAD' or 'MEG MAG'

    #data start time before fixation = -820
    window_start = 920
    window_end = window_start+400
    baseline_start = window_start
    baseline_end = window_end

    path_to_target = join(path, 'SI')
    path_to_nontarget = join(path, 'error')
    target_data = load_data(path_to_target,sensor_type) # trials x time x channel
    nontarget_data = load_data(path_to_nontarget,sensor_type)


    target_data = extract_window(target_data,window_start,window_end,baseline_start,baseline_end)
    nontarget_data = extract_window(nontarget_data,window_start,window_end,baseline_start,baseline_end)
    return target_data, nontarget_data
if __name__== '__main__':
    print('It\'s fun!')