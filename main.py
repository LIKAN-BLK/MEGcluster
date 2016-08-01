from load_data import load_data
from os.path import join
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
from classifier import My_Classifier

def get_data(path):
    path_to_target = join(path, 'em_06_SI')
    path_to_nontarget = join(path, 'em_06_error')
    target_data = load_data(path_to_target)
    nontarget_data = load_data(path_to_nontarget)
    return target_data, nontarget_data




def cv_score(target_data,nontarget_data):
    X = np.concatenate((target_data,nontarget_data),axis=0)
    y = np.hstack((np.ones(target_data.shape[0]),np.zeros(nontarget_data.shape[0])))
    my_clf = My_Classifier()

    scorer = lambda estimator, X, y: roc_auc_score(y,my_clf.predict_proba(X))
    cross_val_score(estimator=my_clf,X=X,y=y,scoring=scorer,cv=2)


if __name__=='__main__':
    path = '../meg_data/'
    target_data, nontarget_data = get_data(path)
    cv_score(target_data,nontarget_data)

