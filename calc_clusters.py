from main import get_data
from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test
import numpy as np
from mne.io import read_raw_fif
from mne.datasets import sample
from mne.channels import find_layout
from mne.viz import plot_topomap
import os

def visualise(data):
    neuromag = read_raw_fif(sample.data_path() + '/MEG/sample/sample_audvis_raw.fif')
    layout = find_layout(neuromag.info, ch_type='grad')
    plot_topomap(data,layout.pos)

def search_clusters(target_data, nontarget_data):
    connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)
    X = [target_data, nontarget_data]
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test(X, n_permutations=1500, connectivity=connectivity[0], check_disjoint=True, tail=0,
                                 n_jobs=8,verbose=True)
    ind = sorted(range(len(cluster_p_values)), key=lambda k: cluster_p_values[k])[:6]
    cluster_sizes = [cluster.size() for cluster in clusters[ind]]
    res = dict(zip(cluster_p_values[ind],cluster_sizes))
    print(res)
    return res

def process_dirs(path_to_experiments):
    entries = os.listdir(path_to_experiments)
    dirs = [os.path.join(path_to_experiments,dir) for dir in entries if os.path.isdir(os.path.join(path_to_experiments,dir))]
    pipe = lambda path: search_clusters(*(get_data(path)))
    result = map(pipe,dirs)
    for i,dir in enumerate(dirs):
        print('%s:' % dir)
        for entry in result[i].items():
            print '  {0}:{1}'.format(*entry)

if __name__ == '__main__':
    process_dirs(os.path.join('..','meg_data'))