from main import get_data
from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test
import numpy as np
import os

def search_clusters(target_data, nontarget_data):
    connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)
    X = [target_data, nontarget_data]
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test(X, n_permutations=1500, connectivity=connectivity[0], check_disjoint=True, tail=0,
                                 n_jobs=8,verbose=True)
    indexes = sorted(range(len(cluster_p_values)), key=lambda k: cluster_p_values[k])[:6]

    cluster_sizes = [clusters[ind].sum() for ind in indexes]
    res = zip(cluster_p_values[indexes],cluster_sizes)
    return res

def process_dirs(path_to_experiments):
    entries = os.listdir(path_to_experiments)
    dirs = [os.path.join(path_to_experiments,dir) for dir in entries if os.path.isdir(os.path.join(path_to_experiments,dir))]
    pipe = lambda path: search_clusters(*(get_data(path)))
    result = map(pipe,dirs)
    for i,dir in enumerate(dirs):
        print('%s:' % dir)
        for entry in result[i]:
            print '  {0}:{1}'.format(*entry)

if __name__ == '__main__':
    process_dirs(os.path.join('..','meg_data'))