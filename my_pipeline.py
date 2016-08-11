
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

class My_pipeline:
    def __init__(self,name,dimension_reduction_step,classifer_step):
        self.name = name
        self.dim_reductuction = copy.deepcopy(dimension_reduction_step)
        self.classification = copy.deepcopy(classifer_step)
        self.scaler = preprocessing.StandardScaler()

    def _apply_cluster_mask(self,X):
        linear_mask = self.cluster_mask.reshape(self.cluster_mask.shape[0]*self.cluster_mask.shape[1])
        data=X.reshape(X.shape[0],X.shape[1] * X.shape[2])[:,np.nonzero(linear_mask)[0]]
        return data

    def fit(self,cluster_mask,X,y):
        self.cluster_mask = cluster_mask
        X = self._apply_cluster_mask(X)
        X = self.scaler.fit_transform(X)
        self.dim_reductuction.fit(X)
        X = self.dim_reductuction.transform(X)

        self.classification.fit(X,y)

    def _transform(self,X):
        X = self._apply_cluster_mask(X)
        X = self.scaler.transform(X)
        return self.dim_reductuction.transform(X)
    def predict(self,X):
        X = self._transform(X)
        return self.classification.predict(X)
    def predict_proba(self,X):
        X = self._transform(X)
        return self.classification.predict_proba(X)
    def score(self,X,y):
        ypred = self.predict_proba(X)[:,1]
        return roc_auc_score(y, ypred, average='macro')

