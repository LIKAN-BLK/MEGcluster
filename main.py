from load_data import load_data
from os.path import join
import numpy as np
from sklearn import cross_validation


import my_pipeline
from sklearn.decomposition import PCA
from lpproj import LocalityPreservingProjection

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import itertools

def get_data(path):
    path_to_target = join(path, 'em_06_SI')
    path_to_nontarget = join(path, 'em_06_error')
    target_data = load_data(path_to_target)
    nontarget_data = load_data(path_to_nontarget)
    return target_data, nontarget_data




def cv_score(target_data,nontarget_data):
    #Cluster methods (now one correct and one empty)
    cluster = my_pipeline.Cluster_test()
    no_cluster = my_pipeline.Empy_cluster_test() #Use OR not to use clasterisation

    # Pool of dimension reduction methods
    pca = PCA(n_components = 60)
    lpp = LocalityPreservingProjection(n_components=30)

    # Pool of classifier methods
    eigen_lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
    lsqr_lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    svd_lda=LinearDiscriminantAnalysis(solver='svd')
    C1_lin_svm = SVC(C=1.0,kernel='linear',gamma='auto',probability=True)
    C1_rbf_svm = SVC(C=1.0,kernel='rbf',gamma='auto',probability=True)
    C0_5_lin_svm = SVC(C=0.5,kernel='linear',gamma='auto',probability=True)
    C0_5_rbf_svm = SVC(C=0.5,kernel='rbf',gamma='auto',probability=True)


    cluster_dict = {'Cluster':cluster,'No_cluster':no_cluster}
    dim_red_dict = {'PCA':pca,'LPP':lpp}
    clf_dict = \
        {'eigen_lda':eigen_lda,'lsqr_lda':lsqr_lda,'svd_lda':svd_lda,'C1_lin_svm':C1_lin_svm,'C1_rbf_svm':C1_rbf_svm,'C0_5_lin_svm':C0_5_lin_svm,'C0_5_rbf_svm':C0_5_rbf_svm}

    my_clf_list = []
    pipe_objects = itertools.product(cluster_dict.values(),dim_red_dict.values(),clf_dict.values(),auc_dict.values())
    pipe_names = itertools.product(cluster_dict.keys(),dim_red_dict.keys(),clf_dict.keys(),auc_dict.keys())
    aucs=[]
    for name,object in zip(pipe_names,pipe_objects):
        aucs[name] = []
        my_clf_list.append(my_pipeline.My_pipeline(name,object))

    X = np.concatenate((target_data,nontarget_data),axis=0)
    y = np.hstack((np.ones(target_data.shape[0]),np.zeros(nontarget_data.shape[0])))

    cv = cross_validation.ShuffleSplit(len(y),n_iter=2,test_size=0.2)


    for train_index,test_index in cv:
        Xtrain = X[train_index, :,:]
        ytrain = y[train_index]

        Xtest = X[test_index,:,:]
        ytest = y[test_index]


        # fit_clf = lambda clf: clf.fit(Xtrain,ytrain)
        {v.fit(Xtrain,ytrain) for v in my_clf_list}

        for v in my_clf_list:
            print
            v.score(Xtest,ytest)



if __name__=='__main__':
    path = '..\\meg_data\\'
    target_data, nontarget_data = get_data(path)
    auc = cv_score(target_data,nontarget_data)
    print('Mean AUC = %f\n' % (auc))
