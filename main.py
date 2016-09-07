from load_data import load_data
import numpy as np
from sklearn import cross_validation
from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test
from mne.time_frequency import cwt_morlet

import my_pipeline
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import itertools
import os
import sys

def get_data(path):
    path_to_target = os.path.join(path, 'SI')
    path_to_nontarget = os.path.join(path, 'error')
    target_data = load_data(path_to_target)
    nontarget_data = load_data(path_to_nontarget)
    return target_data, nontarget_data

def calc_cluster_mask(X,y,freqs):
        def calc_threshold(p_thresh,n_samples_per_group):
            from scipy import stats
            ppf = stats.f.ppf
            p_thresh = p_thresh / 2 # Two tailed
            threshold = ppf(1. - p_thresh, *n_samples_per_group)
            print('P threshold =%f, F threshold = %f' %(p_thresh*2,threshold) )
            return threshold

        p_threshold = 0.00001
        threshold = calc_threshold(p_threshold,[sum(y==1),sum(y==0)])
        connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)

        target_data = X[y==1,:,:,:]
        nontarget_data = X[y==0,:,:,:]

        times_numer = X.shape[1]
        channel_numer = X.shape[2]
        res_clusters = np.empty((times_numer,channel_numer,0))
        freq_number = X.shape[3]

        cluster_threshold = 0.2
        step = freqs[1]-freqs[0]
        for freq_index in xrange(freq_number):
            print('Clustering frequency number %f: freq %f, step %f\n' %(freq_index,freqs[freq_index],step))
            data=[target_data[:,:,:,freq_index],nontarget_data[:,:,:,freq_index]]
            T_obs, clusters, cluster_p_values, H0 = \
                    permutation_cluster_test(data, n_permutations=1500, connectivity=connectivity[0], check_disjoint=True, tail=0,
                                     n_jobs=8,verbose=False,threshold=threshold)


            if any(cluster_p_values < cluster_threshold):
                print('Found clusters lower p=%f' %cluster_threshold)
                for ind_cl, cl in enumerate(clusters):
                    if cluster_p_values[ind_cl] < cluster_threshold:
                        print cluster_p_values[ind_cl],cl.sum()
                res_clusters = np.dstack((res_clusters,reduce(lambda res,x:res | x,[cl for ind_cl,cl in enumerate(clusters) if cluster_p_values[ind_cl]<cluster_threshold])))
            else:

                res_clusters = np.dstack((res_clusters,np.zeros((times_numer,channel_numer))))
                print('Not found valuable cluster, min p = %f\n' %np.min(cluster_p_values))

        return res_clusters


def cv_score(target_data,nontarget_data):
    #Cluster methods (now one correct and one empty)

    # Pool of dimension reduction methods
    pca = PCA(n_components = 60, whiten=True)


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
    freqs = np.arange(15, 25, 2)
    X = np.zeros((source.shape[0],source.shape[2],source.shape[1],len(freqs)))
    for i in xrange(source.shape[0]):
        X[i,:,:,:] = np.log10(np.absolute(cwt_morlet(source[i,:,:], sfreq, freqs, use_fft=True, n_cycles=3.0, zero_mean=True, decim=1))).transpose(2, 0, 1)

    cv = cross_validation.ShuffleSplit(len(y),n_iter=5,test_size=0.2)
    for num_fold,(train_index,test_index) in enumerate(cv):
        print('Fold number %d\n' %(num_fold))
        Xtrain = X[train_index, :,:,:]
        ytrain = y[train_index]

        Xtest = X[test_index,:,:,:]
        ytest = y[test_index]

        cluster_mask = calc_cluster_mask(Xtrain,ytrain,freqs)

        if (cluster_mask.size != 0) & (cluster_mask.any()):
            [v.fit(cluster_mask,Xtrain,ytrain) for v in my_clf_list]

            for v in my_clf_list:
                tmp_auc = v.score(Xtest,ytest)
                print('CLUSTER,dim_reduction=%s,classification=%s, AUC = %f' \
                      %(v.name[0],v.name[1], tmp_auc))
                cluster_aucs[v.name] = np.append(cluster_aucs[v.name],tmp_auc)
        else:
            print('Can not find meaningful cluster')

        [v.fit(np.ones(Xtrain.shape[1:]),Xtrain,ytrain) for v in my_clf_list]

        for v in my_clf_list:
           tmp_auc = v.score(Xtest,ytest)
           print('NO_CLUSTER,dim_reduction=%s,classification=%s, AUC = %f' \
                 %(v.name[0],v.name[1], tmp_auc))
           no_cluster_aucs[v.name] = np.append(no_cluster_aucs[v.name],tmp_auc)

    print('Final results')
    cluster_aucs = {k:v.mean() for k,v in cluster_aucs.items()}
    for k,v in cluster_aucs.items():
        print('CLUSTER,dim_reduction=%s,classification=%s, Mean_AUC = %f' \
                  %(k[0],k[1], v))
    cluster_max_key = max(cluster_aucs,key=cluster_aucs.get)
    print('\nMAX CLUSTER RESULT:CLUSTER,dim_reduction=%s,classification=%s, Mean_AUC = %f\n' \
        %(cluster_max_key[0],cluster_max_key[1], cluster_aucs[cluster_max_key]))

    no_cluster_aucs = {k:v.mean() for k,v in no_cluster_aucs.items()}
    for k,v in no_cluster_aucs.items():
        print('NOCLUSTER,dim_reduction=%s,classification=%s, Mean_AUC = %f' \
                  %(k[0],k[1], v))
    nocluster_max_key = max(no_cluster_aucs,key=no_cluster_aucs.get)
    print('\nMAX NOCLUSTER RESULT:CLUSTER,dim_reduction=%s,classification=%s, Mean_AUC = %f\n' \
                  %(nocluster_max_key[0],nocluster_max_key[1], no_cluster_aucs[nocluster_max_key]))

    if (no_cluster_aucs[nocluster_max_key]>no_cluster_aucs[nocluster_max_key]):
        print('TOTALMAX:NOCLUSTER RESULT:CLUSTER,dim_reduction=%s,classification=%s, Mean_AUC = %f' \
                  %(nocluster_max_key[0],nocluster_max_key[1], no_cluster_aucs[nocluster_max_key]))
    else:
        print('\nMAX CLUSTER RESULT:CLUSTER,dim_reduction=%s,classification=%s, Mean_AUC = %f' \
                  %(cluster_max_key[0],cluster_max_key[1], cluster_aucs[cluster_max_key]))





if __name__=='__main__':
    exp_num=sys.argv[1]
    if not os.path.isdir('results'):
        os.mkdir('results')
    # sys.stdout = open(os.path.join('.','results',exp_num+'.log'), 'w')

    path = os.path.join('..', 'meg_data',exp_num)
    target_data, nontarget_data = get_data(path)
    cv_score(target_data,nontarget_data)
