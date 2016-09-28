from load_data import get_data
# from mne.channels import read_ch_connectivity
# from mne.stats import permutation_cluster_test
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import sys


# connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)
#
# def calc_cluster_mask(X,y):
#         p_threshold = 0.000005 #Magic
#
#         def calc_threshold(p_thresh,n_samples_per_group):
#             from scipy import stats
#             ppf = stats.f.ppf
#             p_thresh = p_thresh / 2 # Two tailed
#             threshold = ppf(1. - p_thresh, *n_samples_per_group)
#             print('P threshold =%f, F threshold = %f' %(p_thresh*2,threshold) )
#             return threshold
#
#         threshold = calc_threshold(p_threshold,[sum(y==1),sum(y==0)])
#         data = [X[y == 1,:,:], X[y == 0,:,:]]
#         T_obs, clusters, cluster_p_values, H0 = \
#                 permutation_cluster_test(data, n_permutations=1000, connectivity=connectivity[0], check_disjoint=True, tail=0,
#                                  threshold=threshold,n_jobs=4,verbose=False)
#         cluster_threshold = 0.2
#         print('Found clusters lower p=%f' %cluster_threshold)
#         for ind_cl, cl in enumerate(clusters):
#             if cluster_p_values[ind_cl] < cluster_threshold:
#                 print cluster_p_values[ind_cl],cl.sum()
#         return reduce(lambda res,x:res | x,[cl for ind_cl,cl in enumerate(clusters) if cluster_p_values[ind_cl]<cluster_threshold])

# def apply_cluster_mask(data,mask):
#     #due to geometry troubles returns data in 2d shape (trials x features)
#     linear_mask = mask.reshape(mask.shape[0]*mask.shape[1])
#     data=data.reshape(data.shape[0],data.shape[1] * data.shape[2])[:,np.nonzero(linear_mask)[0]]
#     return data
#
# def feature_extraction_with_cluster(Xtrain,ytrain,Xtest,ytest):
#
#      cluster_mask = calc_cluster_mask(Xtrain,ytrain)
#      Xtrain = apply_cluster_mask(Xtrain, cluster_mask)
#      effective_pca_num = 160
#      pca = PCA(n_components=effective_pca_num,whiten = True)
#      Xtrain = pca.fit_transform(Xtrain)
#      Xtest = apply_cluster_mask(Xtest, cluster_mask)
#      Xtest=pca.transform(Xtest)
#      return Xtrain,Xtest

def feature_extraction_fullPCA(X_grad_train,X_grad_test,X_mag_train,X_mag_test):
    #Function calculates z-score for each feature (by using mean and variance for this feature in all trials),
    #then concatenates gradiometers and magnitometers and calculates first N principal components on whole dataset

    from sklearn.preprocessing import StandardScaler

    def flat_n_standartize(Xtrain,Xtest):
        # Flatten times x channels arrays and calc z-score
        Xtrain = Xtrain.reshape(Xtrain.shape[0],-1) #flatten array n_samples x n_time x n_channels to n_samples x n_features
        Xtest = Xtest.reshape(Xtest.shape[0],-1)    #Gradiometers (~10^-3) and magnitiometers(~10^-6) have different scale and because we will process them with PCA together we have to make them
        # same scale
        scaler = StandardScaler().fit(Xtrain)
        return scaler.transform(Xtrain),scaler.transform(Xtest)
    X_grad_train,X_grad_test = flat_n_standartize(X_grad_train,X_grad_test)
    X_mag_train,X_mag_test = flat_n_standartize(X_mag_train,X_mag_test)

    effective_pca_num = 160 # PCA components

    # Whitening scales variance to unit, without this svm would not work
    pca = PCA(n_components=effective_pca_num,whiten = True)
    Xtrain = pca.fit_transform(np.hstack((X_grad_train,X_mag_train)))
    Xtest = pca.transform(np.hstack((X_grad_test,X_mag_test)))
    return Xtrain,Xtest

def feature_extraction_partialPCA(X_grad_train,X_grad_test,X_mag_train,X_mag_test):
    #Function flatten data, then center them and calculates PCA on data from each sensor (grad & magn) type separately
    #then standartise them (z-score)

    from sklearn.preprocessing import StandardScaler
    def flat_n_standartize(Xtrain,Xtest):
        # Flatten times x channels arrays and calc z-score
        Xtrain = Xtrain.reshape(Xtrain.shape[0],-1) #flatten array n_samples x n_time x n_channels to n_samples x n_features
        mean = Xtrain.mean(axis=0)
        Xtrain = Xtrain - mean
        Xtest = Xtest.reshape(Xtest.shape[0],-1)
        Xtest = Xtest - mean
        return Xtrain,Xtest #Data with same sensor type have same scale 
    X_grad_train,X_grad_test = flat_n_standartize(X_grad_train,X_grad_test)
    X_mag_train,X_mag_test = flat_n_standartize(X_mag_train,X_mag_test)

    effective_pca_num = 40 # PCA components

    # Whitening scales variance to unit, without this svm would not work
    pca = PCA(n_components=effective_pca_num,whiten = True)
    X_grad_train=pca.fit_transform(X_grad_train)
    X_grad_test=pca.transform(X_grad_test)

    X_mag_train= pca.fit_transform(X_mag_train)
    X_mag_test=pca.transform(X_mag_test)
    Xtrain = np.hstack((X_grad_train,X_mag_train))
    Xtest = np.hstack((X_grad_test,X_mag_test))

    scaler = StandardScaler().fit(Xtrain)
    return scaler.transform(Xtrain),scaler.transform(Xtest)


def cv_score(target_grad_data,nontarget_grad_data,target_mag_data,nontarget_mag_data):

    #initialise classifiers
    eigen_lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto')
    lsqr_lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    svd_lda=LinearDiscriminantAnalysis(solver='svd')
    C10_lin_svm = SVC(C=10,kernel='linear',gamma='auto',probability=True)
    C10_rbf_svm = SVC(C=10,kernel='rbf',gamma='auto',probability=True)
    C1_lin_svm = SVC(C=1.0,kernel='linear',gamma='auto',probability=True)
    C1_rbf_svm = SVC(C=1.0,kernel='rbf',gamma='auto',probability=True)
    C0_1_lin_svm = SVC(C=0.1,kernel='linear',gamma='auto',probability=True)
    C0_1_rbf_svm = SVC(C=0.1,kernel='rbf',gamma='auto',probability=True)

    #dict to to store classifier objects
    clf_dict = \
        {'eigen_lda':eigen_lda,'lsqr_lda':lsqr_lda,'svd_lda':svd_lda,'C0_1_lin_svm':C0_1_lin_svm,'C0_1_rbf_svm':C0_1_rbf_svm,'C1_lin_svm':C1_lin_svm,'C1_rbf_svm':C1_rbf_svm,'C10_lin_svm':C10_lin_svm,'C10_rbf_svm':C10_rbf_svm}

    #dict to to store classifier's auc in each fold
    auc_dict = \
        {'eigen_lda':np.array([]),'lsqr_lda':np.array([]),'svd_lda':np.array([]),'C0_1_lin_svm':np.array([]),'C0_1_rbf_svm':np.array([]),'C1_lin_svm':np.array([]),'C1_rbf_svm':np.array([]),'C10_lin_svm':np.array([]),'C10_rbf_svm':np.array([])}

    #More common view of data for classification X_xx - objects x times x channels, y - lables
    X_grad = np.concatenate((target_grad_data,nontarget_grad_data),axis=0)
    X_mag = np.concatenate((target_mag_data,nontarget_mag_data),axis=0)
    # 1 - label of target class, 0 - label of nontarget class
    y=np.concatenate([np.ones(target_grad_data.shape[0]),np.zeros(nontarget_grad_data.shape[0])])

    #Initialise cross validation
    cv = cross_validation.ShuffleSplit(len(y),n_iter=5,test_size=0.2)
    for train_index,test_index in cv:

        X_grad_train = X_grad[train_index, :,:]
        X_mag_train = X_mag[train_index, :,:]

        ytrain = y[train_index]

        X_grad_test = X_grad[test_index,:,:]
        X_mag_test = X_mag[test_index,:,:]

        ytest = y[test_index]

        Xtrain,Xtest = feature_extraction_fullPCA(X_grad_train,X_grad_test,X_mag_train,X_mag_test)

        # Lambda expression to train each classifier in dict of classifiers
        fit_clf = lambda clf: clf.fit(Xtrain,ytrain)
        {fit_clf(v) for k,v in clf_dict.items()}

        # Lambda expression to train each classifier in dict of classifiers
        calc_auc = lambda clf:roc_auc_score(ytest, clf.predict_proba(Xtest)[:,1], average='macro')
        auc_dict = {k:np.append(v,calc_auc(clf_dict[k])) for k,v in auc_dict.items()}

    #Printing results of classification
    print('Final results')
    mean_aucs = {k:(v.mean(),v.std()) for k,v in auc_dict.items()}
    for k,v in mean_aucs.items():
        print('classification=%s, Mean_AUC = %f, std = %f' \
                  %(k, v[0],v[1]))
    cluster_max_key = max(mean_aucs,key=mean_aucs.get)
    print('\nMAX RESULT:CLUSTER, classification=%s, Mean_AUC = %f, std = %f\n' \
                  %(cluster_max_key, mean_aucs[cluster_max_key][0],mean_aucs[cluster_max_key][1]))
    return mean_aucs


if __name__=='__main__':

    exp_num=sys.argv[1]
    path = join('..', 'meg_data1',exp_num)

    #Loading data
    target_grad_data, nontarget_grad_data = get_data(path,'MEG GRAD')
    # mean(intertrial) oscilates near 10^(-12)
    # variance differ from 10^(-21) to 10^(-22)
    # (mean variance between time x space points 10^(-21)
    target_mag_data, nontarget_mag_data = get_data(path,'MEG MAG')
    # mean(intertrial) oscilates near 10^(-13) and 10^(-14)
    #variance differ from 10^(-23) to 10^(-24)
    # (mean variance between time x space points 10^(-24)
    #Run crossvalidation
    cv_score(target_grad_data,nontarget_grad_data,target_mag_data,nontarget_mag_data)

