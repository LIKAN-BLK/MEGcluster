from load_data import get_data
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



connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)

def calc_cluster_mask(X,y):
        p_threshold = 0.000005 #Magic

        def calc_threshold(p_thresh,n_samples_per_group):
            from scipy import stats
            ppf = stats.f.ppf
            p_thresh = p_thresh / 2 # Two tailed
            threshold = ppf(1. - p_thresh, *n_samples_per_group)
            print('P threshold =%f, F threshold = %f' %(p_thresh*2,threshold) )
            return threshold

        threshold = calc_threshold(p_threshold,[sum(y==1),sum(y==0)])
        data = [X[y == 1,:,:], X[y == 0,:,:]]
        T_obs, clusters, cluster_p_values, H0 = \
                permutation_cluster_test(data, n_permutations=1000, connectivity=connectivity[0], check_disjoint=True, tail=0,
                                 threshold=threshold,n_jobs=4,verbose=False)
        cluster_threshold = 0.2
        print('Found clusters lower p=%f' %cluster_threshold)
        for ind_cl, cl in enumerate(clusters):
            if cluster_p_values[ind_cl] < cluster_threshold:
                print cluster_p_values[ind_cl],cl.sum()
        return reduce(lambda res,x:res | x,[cl for ind_cl,cl in enumerate(clusters) if cluster_p_values[ind_cl]<cluster_threshold])
        # clusters[np.argmin(cluster_p_values)]

def apply_cluster_mask(data,mask):
    #due to geometry troubles returns data in 2d shape (trials x features)
    linear_mask = mask.reshape(mask.shape[0]*mask.shape[1])
    data=data.reshape(data.shape[0],data.shape[1] * data.shape[2])[:,np.nonzero(linear_mask)[0]]
    return data

def feature_extraction_with_cluster(Xtrain,ytrain,Xtest,ytest):

     cluster_mask = calc_cluster_mask(Xtrain,ytrain)
     Xtrain = apply_cluster_mask(Xtrain, cluster_mask)
     effective_pca_num = 80
     pca = PCA(n_components=effective_pca_num,whiten = True)
     Xtrain = pca.fit_transform(Xtrain)
     Xtest = apply_cluster_mask(Xtest, cluster_mask)
     Xtest=pca.transform(Xtest)
     return Xtrain,Xtest

def cv_score(target_data,nontarget_data):

    eigen_lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
    lsqr_lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    svd_lda=LinearDiscriminantAnalysis(solver='svd')
    C1_lin_svm = SVC(C=1.0,kernel='linear',gamma='auto',probability=True)
    C1_rbf_svm = SVC(C=1.0,kernel='rbf',gamma='auto',probability=True)
    C0_5_lin_svm = SVC(C=0.5,kernel='linear',gamma='auto',probability=True)
    C0_5_rbf_svm = SVC(C=0.5,kernel='rbf',gamma='auto',probability=True)
    clf_dict = {'eigen_lda':eigen_lda,'lsqr_lda':lsqr_lda,'svd_lda':svd_lda,'C1_lin_svm':C1_lin_svm,'C1_rbf_svm':C1_rbf_svm,'C0_5_lin_svm':C0_5_lin_svm,'C0_5_rbf_svm':C0_5_rbf_svm}
    auc_dict = {'eigen_lda':np.array([]),'lsqr_lda':np.array([]),'svd_lda':np.array([]),'C1_lin_svm':np.array([]),'C1_rbf_svm':np.array([]),'C0_5_lin_svm':np.array([]),'C0_5_rbf_svm':np.array([])}

    X=np.concatenate((target_data,nontarget_data),axis=0)
    y=np.concatenate([np.ones(target_data.shape[0]),np.zeros(target_data.shape[0])])
    cv = cross_validation.ShuffleSplit(len(y),n_iter=2,test_size=0.2)
    for train_index,test_index in cv:
        Xtrain = X[train_index, :,:]
        ytrain = y[train_index]

        Xtest = X[test_index,:,:]
        ytest = y[test_index]
        Xtrain,Xtest = feature_extraction_with_cluster(Xtrain,ytrain,Xtest,ytest)

        fit_clf = lambda clf: clf.fit(Xtrain,ytrain)
        {fit_clf(v) for k,v in clf_dict.items()}

        calc_auc = lambda clf:roc_auc_score(ytest, clf.predict_proba(Xtest)[:,1], average='macro')
        auc_dict = {k:np.append(v,calc_auc(clf_dict[k])) for k,v in auc_dict.items()}

    auc_dict={k:v.mean() for k,v in auc_dict.items()}
    print auc_dict

if __name__=='__main__':
    path = join('..', 'meg_data1','em06')
    target_data, nontarget_data = get_data(path,'MEG GRAD')
    target_data = target_data[0:75,:]
    nontarget_data = nontarget_data[0:75,:]

    cv_score(target_data,nontarget_data)

