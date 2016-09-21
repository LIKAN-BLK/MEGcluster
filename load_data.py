from scipy.io import loadmat
from os import listdir
from os.path import join
from scipy.signal import butter, lfilter
import numpy as np

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data,cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y=np.zeros(data.shape)
    for tr in range(data.shape[0]):
        for ch in range(data.shape[2]):
            y[tr,:,ch] = lfilter(b, a, data[tr,:,ch])
    return y

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

def extract_window_n_baseline(data,window_start,window_end,baseline_start,baseline_end):
    #Extract window with baseline correction
    target_window = data[:,window_start:window_end:]
    baseline = np.tile(data[:,baseline_start:baseline_end:].mean(axis=1)[:,np.newaxis,:],(1,target_window.shape[1],1))
    return target_window - baseline

def get_data(path,sensor_type):
    # sensor_type -  'MEG GRAD' or 'MEG MAG'
    #data start time before fixation = -820

    window_start = 920
    window_end = window_start+400
    baseline_start = window_start+200
    baseline_end = window_start+300

    path_to_target = join(path, 'SI')
    path_to_nontarget = join(path, 'error')
    target_data = load_data(path_to_target,sensor_type) # trials x time x channel
    nontarget_data = load_data(path_to_nontarget,sensor_type)

    target_data = extract_window_n_baseline(target_data,window_start,window_end,baseline_start,baseline_end)
    nontarget_data = extract_window_n_baseline(nontarget_data,window_start,window_end,baseline_start,baseline_end)
    return target_data, nontarget_data
if __name__== '__main__':
    print('It\'s fun!')