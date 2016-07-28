from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test
import numpy as np

class Classifier:
    def __init__(self):
        self.connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)

    def _calc_cluster_mask(self,target_data, nontarget_data):
        X = [target_data, nontarget_data]
        T_obs, clusters, cluster_p_values, H0 = \
            permutation_cluster_test(X, n_permutations=1000, connectivity=connectivity[0], check_disjoint=True, tail=0,
                                 n_jobs=4)
        return clusters[np.argmin(cluster_p_values)]

    def fit(self,X,y):
        cluster_mask = self._calc_cluster_mask(X[y == 1,:,:], X[y == 0,:,:])
        pca = PCA(n_components=n_comp)
        data = pca.fit_transform(X)

    def predict(self):