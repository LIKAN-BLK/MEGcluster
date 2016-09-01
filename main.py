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
from pathos.multiprocessing import ProcessPool
import time



def get_data(path):
    path_to_target = join(path, 'SI')
    path_to_nontarget = join(path, 'error')
    target_data = load_data(path_to_target)
    nontarget_data = load_data(path_to_nontarget)
    return target_data, nontarget_data

def calc_cluster_mask(target_data, nontarget_data):
    connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)
    X = [target_data, nontarget_data]
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test(X, n_permutations=500, connectivity=connectivity[0], check_disjoint=True, tail=0,
                                 n_jobs=2)
    return clusters[np.argmin(cluster_p_values)]

def apply_cluster_mask(data,mask):
    #due to geometry troubles returns data in 2d shape (trials x features)
    linear_mask = mask.reshape(mask.shape[0]*mask.shape[1])
    data=data.reshape(data.shape[0],data.shape[1] * data.shape[2])[:,np.nonzero(linear_mask)[0]]
    return data

def cv_fold(fold_number,Xtrain,Xtest,ytrain,ytest,clf):
        print('Fold number %d' %fold_number)
        cluster_mask = calc_cluster_mask(Xtrain[ytrain == 1,:,:], Xtrain[ytrain == 0,:,:])
        Xtrain = apply_cluster_mask(Xtrain, cluster_mask)

        pca = PCA(n_components = min([200,Xtrain.shape[1]]))
        Xtrain = pca.fit_transform(Xtrain)
        print('Real num components = %d' %(Xtrain.shape[1]))

        Xtest = apply_cluster_mask(Xtest, cluster_mask)
        Xtest = pca.transform(Xtest)

        clf.fit(Xtrain,ytrain)
        return roc_auc_score(ytest, clf.predict_proba(Xtest)[:,1], average='macro')



def fold_helper(args):
    import main
    return main.cv_fold(*args)

def cv_score(target_data,nontarget_data):


    clf=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
    X=np.concatenate((target_data,nontarget_data),axis=0)
    y=np.concatenate([np.ones(target_data.shape[0]),np.zeros(target_data.shape[0])])
    cv = cross_validation.ShuffleSplit(len(y),n_iter=4,test_size=0.2)
    acc = []
    auc = []

    arg_list = [(fold_number,X[train_index, :,:],X[test_index, :,:],y[train_index],y[test_index],clf) for fold_number,(train_index,test_index) in enumerate(cv)]

    pool = ProcessPool(processes=1)
    auc = pool.map(fold_helper, arg_list)
    print np.array(auc).mean()

if __name__=='__main__':
    start = time.time()
    path = join('..', 'meg_data','em06')
    target_data, nontarget_data = get_data(path)
    cv_score(target_data,nontarget_data)
    end = time.time()
    print('Time elapsed %d' %(end - start))
