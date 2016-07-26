from load_data import load_data
from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

def get_data(path):
    path_to_target = join(path, 'em_06_SI')
    path_to_nontarget = join(path, 'em_06_error')
    target_data = load_data(path_to_target)
    nontarget_data = load_data(path_to_nontarget)
    return target_data, nontarget_data

def calc_cluster_mask(target_data, nontarget_data):
    connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)
    X = [target_data, nontarget_data]
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test(X, n_permutations=1000, connectivity=connectivity[0], check_disjoint=True, tail=0,
                                 n_jobs=4)
    return clusters[np.argmin(cluster_p_values)]

def apply_cluster_mask(target_data, nontarget_data,mask):
    linear_mask = mask.reshape(mask.shape[0]*mask.shape[1])
    target=target_data.reshape(target_data.shape[0],target_data.shape[1] * target_data.shape[2])[:,np.nonzero(linear_mask)]
    nontarget=nontarget_data.reshape(nontarget_data.shape[0],nontarget_data.shape[1] * nontarget_data.shape[2])[:,np.nonzero(linear_mask)]
    return target,nontarget



def dimension_reduction(target_data,nontarget_data, n_comp=200):
    pca = PCA(n_components=n_comp)
    data = np.concatenate((target_data,nontarget_data),axis=0)
    pca.fit(data)
    target = pca.fit_transform(target_data)
    non_target = pca.fit_transform(nontarget_data)
    return target,non_target


def classification(target_data,nontarget_data):
    clf=LinearDiscriminantAnalysis(solver='eig',shrinkage='auto')
    X=np.concatenate((target_data,nontarget_data),axis=0)
    y=np.concatenate([np.ones(target_data.shape[0]),np.zeros(target_data.shape[0])])
    cross_validation.cross_val_score(clf, X, y,cv=5, scoring='roc_auc')


if __name__=='__main__':
    path_to_target = '../meg_data/em_06_SI'
    path_to_nontarget = '../meg_data/em_06_error'
    target_data =  load_data(path_to_target)
    nontarget_data =  load_data(path_to_nontarget)
    connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)
    X=[target_data,nontarget_data]
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test(X, n_permutations=1000, connectivity=connectivity[0],check_disjoint=True, tail=0, n_jobs=4)

    print('Be happy!')



    times = range(len(target_data[0][:][0]))
    num_of_useful_clust = sum(cluster_p_values < 0.05)


    plt.close('all')
    for i_c, c in enumerate(clusters):
        if cluster_p_values[i_c] <= 0.05:
            plt.imshow(c)
            plt.xlabel('channels')
            plt.ylabel('times')
            plt.title('Cluster p-value = %f\n' % cluster_p_values)
            plt.show()

    # for i_c, c in enumerate(clusters):
    #     c = c[0]
    #     if cluster_p_values[i_c] <= 0.05:
    #         plt.imshow(c)
