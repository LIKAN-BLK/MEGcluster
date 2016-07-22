from load_data import load_data
from mne.channels import read_ch_connectivity
from mne.stats import permutation_cluster_test
import matplotlib.pyplot as plt

if __name__=='__main__':
    path_to_target = '..\\meg_data\\em_06_SI'
    path_to_nontarget = '..\\meg_data\\em_06_error'
    target_data =  load_data(path_to_target)
    nontarget_data =  load_data(path_to_nontarget)
    connectivity = read_ch_connectivity('neuromag306planar_neighb.mat', picks=None)
    X=[target_data,nontarget_data]
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test(X, n_permutations=1000, connectivity=connectivity[0],check_disjoint=True, tail=0, n_jobs=8)

    print('Be happy!')



    times = range(len(target_data[0][:][0]))
    num_of_useful_clust = sum(cluster_p_values < 0.05)


    plt.close('all')
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            plt.imshow(c)
            plt.xlabel('channels')
            plt.ylabel('times')
            plt.title('Cluster p-value = %f\n' % cluster_p_values)
            plt.show()

    # for i_c, c in enumerate(clusters):
    #     c = c[0]
    #     if cluster_p_values[i_c] <= 0.05:
    #         plt.imshow(c)
