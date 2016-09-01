from main import get_data
from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test
import numpy as np
from mne.io import read_raw_fif
from mne.datasets import sample
from mne.channels import find_layout
from mne.viz import plot_topomap
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


info = read_raw_fif(sample.data_path() + '/MEG/sample/sample_audvis_raw.fif',verbose=False).info
connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y=np.zeros(data.shape)
    for tr in range(data.shape[0]):
        for ch in range(data.shape[2]):
            y[tr,:,ch] = lfilter(b, a, data[tr,:,ch])
    return y

def tmp_plot(data):
    import matplotlib.pyplot as plt
    for ch in range(1,11):
        plt.subplot(5, 2, ch)
        plt.plot(data[0,:,ch])


def save_fig(exp_num,title,fig):
    if not os.path.isdir(os.path.join('results',exp_num)):
        os.mkdir(os.path.join('results',exp_num))
    fig.savefig(os.path.join('results',exp_num,title + '.png'))

def visualise(exp_num,main_title,*args):
    layout = find_layout(info, ch_type='grad')
    number_of_heads = len(args)
    tmp = [list(l) for l in zip(*args)]
    titles = tmp[0]
    data = tmp[1]
    max_row_lenght = 5 #depends from monitor length (:
    fig,axes=plt.subplots(-(-number_of_heads//max_row_lenght),min(max_row_lenght,number_of_heads),figsize=(20, 20))
    fig.suptitle(main_title, fontsize=14)
    min_value = np.array(map(lambda x:x.min(),data)).min()
    max_value = np.array(map(lambda x:x.max(),data)).max()
    for i in range(number_of_heads):
        axes[np.unravel_index(i,axes.shape)].set_title(titles[i])
        if data[i].any():
            im,_ = plot_topomap(data[i],layout.pos,axes=axes[np.unravel_index(i,axes.shape)],
                                vmin=min_value,vmax=max_value,show=False)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.3,fraction=0.025)
    save_fig(exp_num,main_title,fig)


def f_heads(exp_num,X):
    from mne.stats.parametric import _f_oneway

    n_times = X[0].shape[1]
    sample_shape = X[0].shape[1:]
    X = [np.reshape(x, (x.shape[0], -1)) for x in X]
    f_obs,p_obs = _f_oneway(*X)
    print('stat_fun(H1): min=%f max=%f' % (np.min(f_obs), np.max(f_obs)))
    f_obs.shape = sample_shape
    p_obs.shape = sample_shape
    step = 50 # ms
    times = range(0,n_times,step)
    f_list = [('[%d:%d]ms' %(times[ind],times[ind+1]),f_obs[times[ind]:times[ind+1],:].mean(axis=0)) for ind in range(len(times)-1)]
    visualise(exp_num,'F values',*f_list)


def clustermask_heads(exp_num,title, cluster_mask):
    n_times = cluster_mask.shape[0]
    step = 50 # ms
    times = range(0,n_times,step)
    cm_list = [('[%d:%d]ms' %(times[ind],times[ind+1]),cluster_mask[times[ind]:times[ind+1],:].mean(axis=0)) for ind in range(len(times)-1)]
    visualise(exp_num,title,*cm_list)


def calc_threshold(p_thresh,n_samples_per_group):
    from scipy import stats
    ppf = stats.f.ppf
    p_thresh = p_thresh / 2 # Two tailed
    threshold = ppf(1. - p_thresh, *n_samples_per_group)
    print('P threshold =%f, F threshold = %f' %(p_thresh*2,threshold) )
    return threshold


def search_clusters(exp_num,target_data, nontarget_data):
    print(exp_num)
    order = 6
    fs = 1000.0       # sample rate, Hz
    cutoff = 25  # desired cutoff frequency of the filter, Hz
    target_data = butter_lowpass_filter(target_data, cutoff, fs, order)
    nontarget_data = butter_lowpass_filter(nontarget_data, cutoff, fs, order)

    X = [target_data, nontarget_data]

    f_heads(exp_num,X)
    p_thresholds = [0.00001,0.000005] #Magic
    for p_threshold in p_thresholds:
        threshold = calc_threshold(p_threshold,[len(target_data),len(nontarget_data)])
        T_obs, clusters, cluster_p_values, H0 = \
            permutation_cluster_test(X, n_permutations=1500, connectivity=connectivity[0],threshold = threshold,
                                     check_disjoint=True, tail=0,n_jobs=6,verbose=False)


        indexes = sorted(range(len(cluster_p_values)), key=lambda k: cluster_p_values[k])[:5]
        for i in indexes:
            if cluster_p_values[i] < 0.2:
                clustermask_heads(exp_num,'CM_thr=%f_p=%f' % (p_threshold,cluster_p_values[i]),clusters[i])
        cluster_sizes = [clusters[ind].sum() for ind in indexes]
        res = zip(cluster_p_values[indexes],cluster_sizes)
        print(res)
        f=open(os.path.join('results' 'res_file.txt'),'a+')
        f.write('%s: Threshold=%f, clusters = %s' % (exp_num, p_threshold,str(res)))
        f.close()
    return res

def process_dirs(path_to_experiments):
    if not os.path.isdir('results'):
        os.mkdir('results')
    entries = os.listdir(path_to_experiments)
    dirs = [dir for dir in entries if os.path.isdir(os.path.join(path_to_experiments,dir))]

    pipe = lambda dir: search_clusters(dir,*(get_data(os.path.join(path_to_experiments,dir))))
    result = map(pipe,dirs)
    for i,dir in enumerate(dirs):
        print('%s:' % dir)
        for entry in result[i]:
            print '  {0}:{1}'.format(*entry)

if __name__ == '__main__':
    f=open(os.path.join('results' 'res_file.txt'),'w+')
    f.close()
    process_dirs(os.path.join('..','meg_data'))
    # test_list = [('A',np.random.rand(204)),('B',np.random.rand(204)),('C',np.random.rand(204))]
    # visualise('Ceep calm, this is just a test',*test_list)
    # plt.show()
    # tX,ntX = get_data(os.path.join('..', 'meg_data','em06'))
    # f_heads([tX,ntX])
    plt.show()

