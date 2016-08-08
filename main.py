from load_data import load_data
from os.path import join
import numpy as np
from sklearn import cross_validation
from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test


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

def calc_cluster_mask(X,y):
        connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)
        data = [X[y == 1,:,:], X[y == 0,:,:]]
        T_obs, clusters, cluster_p_values, H0 = \
                permutation_cluster_test(data, n_permutations=1500, connectivity=connectivity[0], check_disjoint=True, tail=0,
                                 n_jobs=8)
        if any(cluster_p_values < 0.05):
            return clusters[np.argmin(cluster_p_values)]
        else:
            return None


def cv_score(target_data,nontarget_data):
    #Cluster methods (now one correct and one empty)

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


    dim_red_dict = {'PCA':pca,'LPP':lpp}
    clf_dict = \
        {'eigen_lda':eigen_lda,'lsqr_lda':lsqr_lda,'svd_lda':svd_lda,'C1_lin_svm':C1_lin_svm,'C1_rbf_svm':C1_rbf_svm,'C0_5_lin_svm':C0_5_lin_svm,'C0_5_rbf_svm':C0_5_rbf_svm}


    pipe_objects = itertools.product(dim_red_dict.values(),clf_dict.values())
    pipe_names = itertools.product(dim_red_dict.keys(),clf_dict.keys())
    cluster_aucs={}
    no_cluster_aucs={}
    my_clf_list=[]
    for name,object in zip(pipe_names,pipe_objects):
        cluster_aucs[name] = np.array([])
        no_cluster_aucs[name] = np.array([])
        my_clf_list.append(my_pipeline.My_pipeline(name,object[0],object[1]))

    X = np.concatenate((target_data,nontarget_data),axis=0)
    y = np.hstack((np.ones(target_data.shape[0]),np.zeros(nontarget_data.shape[0])))

    cv = cross_validation.ShuffleSplit(len(y),n_iter=2,test_size=0.2)


    for num_fold,(train_index,test_index) in enumerate(cv):
        print('Fold number %d\n' %(num_fold))
        Xtrain = X[train_index, :,:]
        ytrain = y[train_index]

        Xtest = X[test_index,:,:]
        ytest = y[test_index]
        cluster_mask = calc_cluster_mask(Xtrain,ytrain)

        if cluster_mask != None:
            {v.fit(cluster_mask,Xtrain,ytrain) for v in my_clf_list}

            for v in my_clf_list:
                tmp_auc = v.score(Xtest,ytest)
                print('CLUSTER,dim_reduction=%s,classification=%s, AUC = %f' \
                      %(v.name[0],v.name[1], tmp_auc))
                np.append(cluster_aucs[v.name],tmp_auc)
        else:
            print('Can not find meaningful cluster')

        {v.fit(np.ones(Xtrain.shape),Xtrain,ytrain) for v in my_clf_list}

        for v in my_clf_list:
            tmp_auc = v.score(Xtest,ytest)
            print('NO_CLUSTER,dim_reduction=%s,classification=%s, AUC = %f' \
                  %(v.name[0],v.name[1], tmp_auc))
            np.append(no_cluster_aucs[v.name],tmp_auc)

    print('Final results')
    for k,v in cluster_aucs.items():
        print('CLUSTER,dim_reduction=%s,classification=%s, Mean_AUC = %f' \
                  %(v.name[0],v.name[1], tmp_auc.mean()))

    for k,v in no_cluster_aucs.items():
        print('NOCLUSTER,dim_reduction=%s,classification=%s, Mean_AUC = %f' \
                  %(v.name[0],v.name[1], tmp_auc.mean()))




if __name__=='__main__':
    path = '..\\meg_data\\'
    target_data, nontarget_data = get_data(path)
    auc = cv_score(target_data,nontarget_data)
    print('Mean AUC = %f\n' % (auc))
