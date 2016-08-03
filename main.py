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
from sklearn.svm import SVC


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
                                 n_jobs=6)
    return clusters[np.argmin(cluster_p_values)]

def apply_cluster_mask(data,mask):
    #due to geometry troubles returns data in 2d shape (trials x features)
    linear_mask = mask.reshape(mask.shape[0]*mask.shape[1])
    data=data.reshape(data.shape[0],data.shape[1] * data.shape[2])[:,np.nonzero(linear_mask)[0]]
    return data



def cv_score(target_data,nontarget_data):


    eigen_lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
    lsqr_lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    svd_lda=LinearDiscriminantAnalysis(solver='svd')
    C1_lin_svm = SVC(C=1.0,kernel='linear',gamma='auto',probability=True)
    C1_rbf_svm = SVC(C=1.0,kernel='rbf',gamma='auto',probability=True)
    C0_5_lin_svm = SVC(C=0.5,kernel='linear',gamma='auto',probability=True)
    C0_5_rbf_svm = SVC(C=0.5,kernel='rbf',gamma='auto',probability=True)
    clf_dict = {'eigen_lda':eigen_lda,'lsqr_lda':lsqr_lda,'svd_lda':svd_lda,'C1_lin_svm':C1_lin_svm,'C1_rbf_svm':C1_rbf_svm,'C0_5_lin_svm':C0_5_lin_svm,'C0_5_rbf_svm':C0_5_rbf_svm}
    auc_dict = {'eigen_lda':np.array([]),'lsqr_lda':np.array([]),'svd_lda':np.array([]),'C1_lin_svm':np.array([]),'C1_rbf_svm':np.array([]),'C0_5_lin_svm':np.array([]),'C0_5_lin_svm':np.array([])}

    X=np.concatenate((target_data,nontarget_data),axis=0)
    y=np.concatenate([np.ones(target_data.shape[0]),np.zeros(target_data.shape[0])])
    cv = cross_validation.ShuffleSplit(len(y),n_iter=2,test_size=0.5)
    for train_index,test_index in cv:
        Xtrain = X[train_index, :,:]
        ytrain = y[train_index]

        Xtest = X[test_index,:]
        ytest = y[test_index]

        cluster_mask = calc_cluster_mask(Xtrain[ytrain == 1,:,:], Xtrain[ytrain == 0,:,:])
        Xtrain = apply_cluster_mask(Xtrain, cluster_mask)


        pca = PCA(n_components = min([200,Xtrain.shape[1]]))
        Xtrain = pca.fit_transform(Xtrain)
        print('Real num components = %d' %(Xtrain.shape[1]))

        fit_clf = lambda clf: clf.fit(Xtrain,ytrain)
        {fit_clf(v) for k,v in clf_dict.items()}

        Xtest = apply_cluster_mask(Xtest, cluster_mask)
        Xtest = pca.transform(Xtest)

        calc_auc = lambda clf:roc_auc_score(ytest, clf.predict_proba(Xtest)[:,1], average='macro')
        auc_dict = {k:np.append(v,calc_auc(clf_dict[k])) for k,v in auc_dict.items()}

    auc_dict={k:v.mean() for k,v in auc_dict.items()}
    print auc_dict

if __name__=='__main__':
    path = '../meg_data/'
    target_data, nontarget_data = get_data(path)

    cv_score(target_data,nontarget_data)

