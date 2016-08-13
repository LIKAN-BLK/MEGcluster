
import numpy as np
import copy
from sklearn.metrics import roc_auc_score

class My_pipeline:
    def __init__(self,name,dimension_reduction_step,classifer_step):
        self.name = name
        self.dim_reductuction = copy.deepcopy(dimension_reduction_step)
        self.classification = copy.deepcopy(classifer_step)

    def _apply_cluster_mask(self,X):
        linear_mask = np.ndarray.flatten(self.cluster_mask)
        data=X.reshape(X.shape[0],-1)[:,np.nonzero(linear_mask)[0]]
        return data

    def fit(self,cluster_mask,X,y):
        self.cluster_mask = cluster_mask
        X = self._apply_cluster_mask(X)
        self.dim_reductuction.fit(X)
        X = self.dim_reductuction.transform(X)

        self.classification.fit(X,y)

    def predict(self,X):
        X = self._apply_cluster_mask(X)
        X = self.dim_reductuction.transform(X)
        return self.classification.predict(X)
    def predict_proba(self,X):
        X = self._apply_cluster_mask(X)
        X = self.dim_reductuction.transform(X)
        return self.classification.predict_proba(X)
    def score(self,X,y):
        ypred = self.predict_proba(X)[:,1]
        return roc_auc_score(y, ypred, average='macro')

