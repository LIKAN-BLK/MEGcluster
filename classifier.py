from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class My_Classifier:
    def __init__(self):
        self.connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)

    def _apply_cluster_mask(self,data):
            #due to geometry troubles returns data in 2d shape (trials x features)
            linear_mask = self.cluster_mask.reshape(self.cluster_mask.shape[0]*self.cluster_mask.shape[1])
            data=data.reshape(data.shape[0],data.shape[1] * data.shape[2])[:,np.nonzero(linear_mask)[0]]
            return data

    def fit(self,X,y):
        def _calc_cluster_mask(self,target_data, nontarget_data):
            X = [target_data, nontarget_data]
            T_obs, clusters, cluster_p_values, H0 = \
                permutation_cluster_test(X, n_permutations=1000, connectivity=self.connectivity[0], check_disjoint=True, tail=0,
                                 n_jobs=4)
            return clusters[np.argmin(cluster_p_values)]


        clf=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
        self.cluster_mask = _calc_cluster_mask(X[y == 1,:,:], X[y == 0,:,:])

        X = self._apply_cluster_mask(self,X)
        n_comp = min([200,X.shape[0],X.shape[1]])
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(X)
        X = self.pca.transform(X)

        self.clf=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
        self.clf.fit(X,y)

    def predict(self,X):
        X = self._apply_cluster_mask(self,X)
        X = self.pca.transform(X)
        return self.clf.predict(X)
    def predict_proba(self,X):
        X = self._apply_cluster_mask(self,X)
        X = self.pca.transform(X)
        return self.clf.predict_proba(X)

