from scipy.io import loadmat
import numpy as np
from os import listdir
from os.path import join
from mne.time_frequency import cwt_morlet

def load_data(path):
    mask_rawdata = loadmat('ChannelType.mat')
    gradiom_mask = np.array(map(lambda x: x[0][0] == 'MEG GRAD',mask_rawdata['Type']))
    return np.concatenate([extract_grad_mat(join(path,f),gradiom_mask) for f in listdir(path) if f.endswith(".mat")],axis=0) #By default trial x channel x time


def extract_grad_mat(path,gradiom_mask):
    data=loadmat(path)
    return (data['F'][gradiom_mask])[np.newaxis,...] #additional dimension for easier concatenation to 3d array in the future


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
    target_data = load_data(path_to_target)
    nontarget_data = load_data(path_to_nontarget)

    target_data = tft(target_data,window_start,window_end,baseline_start,baseline_end)
    nontarget_data = tft(nontarget_data,window_start,window_end,baseline_start,baseline_end)
    return target_data, nontarget_data

def tft(source,window_start,window_end,baseline_start,baseline_end):
    sfreq=1000 #Sampling freq 1000Hz
    freqs = np.arange(15, 25, 2)
    res = np.zeros((source.shape[0],(window_end-window_start),source.shape[1],len(freqs)))
    for i in xrange(source.shape[0]):
        tf_magnitude = np.absolute(cwt_morlet(source[i,:,:window_end], sfreq, freqs, use_fft=True, n_cycles=3.0, zero_mean=True, decim=1))
        tf_magnitude_baseline = tf_magnitude[:,:,baseline_start:baseline_end].mean(axis=2)
        tf_magnitude = tf_magnitude[:,:,window_start:]
        tf_magnitude = np.log10(tf_magnitude) - np.log10(np.tile(tf_magnitude_baseline[:,:,np.newaxis],(1,1,tf_magnitude.shape[2])))
        res[i,:,:,:] = (tf_magnitude).transpose(2, 0, 1)
    return res


if __name__== '__main__':
    print('It\'s fun!')