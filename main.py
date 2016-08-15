from load_data import load_data
from os.path import join
import numpy as np
from sklearn import cross_validation
from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test
from mne.time_frequency import cwt_morlet

import my_pipeline
from sklearn.decomposition import PCA
# from lpproj import LocalityPreservingProjection

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
        # target_indexes = np.nonzero(y==1)
        # nontarget_indexes = np.nonzero(y==0)
        target_data = X[y==1,:,:,:]
        nontarget_data = X[y==0,:,:,:]

        times_numer = X.shape[1]
        channel_numer = X.shape[2]
        res_clusters = np.empty((times_numer,channel_numer,0))
        freq_number = X.shape[3]
        for freq_index in xrange(freq_number):
            print('Clustering frequency number %f\n' %freq_index)
            data=[target_data[:,:,:,freq_index],nontarget_data[:,:,:,freq_index]]
            T_obs, clusters, cluster_p_values, H0 = \
                    permutation_cluster_test(data, n_permutations=500, connectivity=connectivity[0], check_disjoint=True, tail=0,
                                     n_jobs=8,verbose=False)
            if any(cluster_p_values < 0.5):
                res_clusters = np.dstack((res_clusters,clusters[np.argmin(cluster_p_values)]))
                print('Found valuable cluster, p = %f\n' %np.min(cluster_p_values))
            else:
                
                res_clusters = np.dstack((res_clusters,np.zeros((times_numer,channel_numer))))
                print('Not found valuable cluster, min p = %f\n' %np.min(cluster_p_values))

        return res_clusters


def cv_score(target_data,nontarget_data):
    #Cluster methods (now one correct and one empty)

    # Pool of dimension reduction methods
    pca = PCA(n_components = 60, whiten=True)
    # lpp = LocalityPreservingProjection(n_components=30)

    # Pool of classifier methods
    eigen_lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
    lsqr_lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    svd_lda=LinearDiscriminantAnalysis(solver='svd')
    C10_lin_svm = SVC(C=10,kernel='linear',gamma='auto',probability=True)
    C10_rbf_svm = SVC(C=10,kernel='rbf',gamma='auto',probability=True)
    C1_lin_svm = SVC(C=1.0,kernel='linear',gamma='auto',probability=True)
    C1_rbf_svm = SVC(C=1.0,kernel='rbf',gamma='auto',probability=True)
    C0_1_lin_svm = SVC(C=0.1,kernel='linear',gamma='auto',probability=True)
    C0_1_rbf_svm = SVC(C=0.1,kernel='rbf',gamma='auto',probability=True)


    dim_red_dict = {'PCA':pca}
    clf_dict = \
        {'eigen_lda':eigen_lda,'lsqr_lda':lsqr_lda,'svd_lda':svd_lda,'C10_lin_svm':C10_lin_svm,'C10_rbf_svm':C10_rbf_svm,'C1_lin_svm':C1_lin_svm,'C1_rbf_svm':C1_rbf_svm,'C0_1_lin_svm':C0_1_lin_svm,'C0_1_rbf_svm':C0_1_rbf_svm}


    pipe_objects = itertools.product(dim_red_dict.values(),clf_dict.values())
    pipe_names = itertools.product(dim_red_dict.keys(),clf_dict.keys())
    cluster_aucs={}
    no_cluster_aucs={}
    my_clf_list=[]
    for name,object in zip(pipe_names,pipe_objects):
        cluster_aucs[name] = np.array([])
        no_cluster_aucs[name] = np.array([])
        my_clf_list.append(my_pipeline.My_pipeline(name,object[0],object[1]))


    source = np.concatenate((target_data,nontarget_data),axis=0)
    y = np.hstack((np.ones(target_data.shape[0]),np.zeros(nontarget_data.shape[0])))



    sfreq=1000 #Sampling freq 1000Hz
    freqs = np.arange(10, 81, 20)
    X = np.zeros((source.shape[0],source.shape[2],source.shape[1],len(freqs)))
    for i in xrange(source.shape[0]):
        X[i,:,:,:] = np.log10(np.absolute(cwt_morlet(source[i,:,:], sfreq, freqs, use_fft=True, n_cycles=2.0, zero_mean=False, decim=1))).transpose(2, 0, 1)

    cv = cross_validation.ShuffleSplit(len(y),n_iter=5,test_size=0.2)
    for num_fold,(train_index,test_index) in enumerate(cv):
        print('Fold number %d\n' %(num_fold))
        Xtrain = X[train_index, :,:,:]
        ytrain = y[train_index]

        Xtest = X[test_index,:,:,:]
        ytest = y[test_index]

        cluster_mask = calc_cluster_mask(Xtrain,ytrain)

        if cluster_mask.size != 0:
            [v.fit(cluster_mask,Xtrain,ytrain) for v in my_clf_list]

            for v in my_clf_list:
                tmp_auc = v.score(Xtest,ytest)
                print('CLUSTER,dim_reduction=%s,classification=%s, AUC = %f' \
                      %(v.name[0],v.name[1], tmp_auc))
                cluster_aucs[v.name] = np.append(cluster_aucs[v.name],tmp_auc)
        else:
            print('Can not find meaningful cluster')

        #[v.fit(np.ones(Xtrain.shape[1:]),Xtrain,ytrain) for v in my_clf_list]

        #for v in my_clf_list:
        #    tmp_auc = v.score(Xtest,ytest)
        #    print('NO_CLUSTER,dim_reduction=%s,classification=%s, AUC = %f' \
        #          %(v.name[0],v.name[1], tmp_auc))
        #    no_cluster_aucs[v.name] = np.append(no_cluster_aucs[v.name],tmp_auc)

    print('Final results')
    for k,v in cluster_aucs.items():
        print('CLUSTER,dim_reduction=%s,classification=%s, Mean_AUC = %f' \
                  %(k[0],k[1], v.mean()))

    #for k,v in no_cluster_aucs.items():
    #    print('NOCLUSTER,dim_reduction=%s,classification=%s, Mean_AUC = %f' \
    #              %(k[0],k[1], v.mean()))




if __name__=='__main__':
    path = join('..', 'meg_data')
    target_data, nontarget_data = get_data(path)
    cv_score(target_data,nontarget_data)
