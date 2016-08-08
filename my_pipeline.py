from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test
import numpy as np
import copy
from sklearn.metrics import roc_auc_score

class My_pipeline:
    def __init__(self,name,cluster_step,dimension_reduction_step,classifer_step):
        self.name = name
        self.clusterization = copy.deepcopy(cluster_step)
        self.dim_reductuction = copy.deepcopy(dimension_reduction_step)
        self.classification = copy.deepcopy(classifer_step)


    def fit(self,X,y):
        self.clusterization.fit(X,y)
        X = self.clusterization.transform(X)
        X = self.dim_reductuction.fit(X)
        X = self.dim_reductuction.transform(X)

        self.classification.fit(X,y)

    def predict(self,X):
        X = self.clusterization.transform(X)
        X = self.dim_reductuction.transform(X)
        return self.classification.predict(X)
    def predict_proba(self,X):
        X = self.clusterization.transform(X)
        X = self.dim_reductuction.transform(X)
        return self.classification.predict_proba(X)
    def score(self,X,y):
        ypred = self.predict_proba(X)[:,1]
        return roc_auc_score(y, ypred, average='macro')


class Cluster_test:
    def __init__(self):
        self.connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)
    def fit(self,X,y):
        data = [X[y == 1,:,:], X[y == 0,:,:]]
        T_obs, clusters, cluster_p_values, H0 = \
                permutation_cluster_test(data, n_permutations=1000, connectivity=self.connectivity[0], check_disjoint=True, tail=0,
                                 n_jobs=8)
        if any(cluster_p_values < 0.05):
            self.mask = clusters[np.argmin(cluster_p_values)]
        else:
            self.mask = None
    def transform(self,X):
        linear_mask = self.mask.reshape(self.mask.shape[0]*self.mask.shape[1])
        data=X.reshape(X.shape[0],X.shape[1] * X.shape[2])[:,np.nonzero(linear_mask)[0]]
        return data


class Empy_cluster_test:
    def __init__(self):
        pass
    def fit(self,X,y):
        pass
    def transform(self,X):
        data=X.reshape(X.shape[0],X.shape[1] * X.shape[2])
        return data
