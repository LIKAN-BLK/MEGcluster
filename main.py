from load_data import load_data
from os.path import join
import numpy as np
from sklearn import cross_validation
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
    cv = cross_validation.ShuffleSplit(len(y),n_iter=2,test_size=0.2)
    auc=[]
    for train_index,test_index in cv:
        Xtrain = X[train_index, :,:]
        ytrain = y[train_index]

        Xtest = X[test_index,:,:]
        ytest = y[test_index]

        my_clf.fit(Xtrain,ytrain)
        ypred = my_clf.predict_proba(Xtest)[:,1]
        fold_auc=roc_auc_score(ytest, ypred, average='macro')
        print('Mean AUC = %f\n' % (fold_auc))
        auc.append(fold_auc)
    return sum(auc)/float(len(auc))



if __name__=='__main__':
    path = '..\\meg_data\\'
    target_data, nontarget_data = get_data(path)
    auc = cv_score(target_data,nontarget_data)
    print('Mean AUC = %f\n' % (auc))
