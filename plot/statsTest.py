import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# feature statistic analysis

rootdir = '/home/cjy/data/10.11'
fold = 'raw'
featuredir = 'feature'

fdir = os.path.join(rootdir, fold, featuredir)

mod = ['DWI', 'T1+c', 'T2', 'T2-FLAIR']
slice_oris = ['transverse', 'sagittal', 'coronal']

p_result = np.zeros((16, 12))

for index in range(4):
    for slice_ori in range(3):
        # X = np.load('feature/v5/X' + mod[index] + '_' + slice_oris[slice_ori] + '.npy')
        X = np.load(os.path.join(fdir, mod[index] + '_' + slice_oris[slice_ori] + '.npy'))

        s = []
        p = []
        for i in range(X.shape[1]):
            # group a: meta
            a = X[0:45, i]
            # group b : gbm
            b = X[45:88, i]
            # plot data feature distribution between groups
            data_a = pd.DataFrame(a, columns=['meta'])
            data_b = pd.DataFrame(b, columns=['gbm'])
            plt.boxplot([data_a, data_b])

            # plt.savefig('plot/v6/' + mod[index] + '_' + slice_oris[slice_ori] + '_' + str(i) + '.jpg')
            plt.savefig(os.path.join(fdir, 'stats', mod[index] + '_' + slice_oris[slice_ori] + '_' + str(i) + '.jpg'))
            plt.show()

            u, pvalue = stats.mannwhitneyu(a, b, alternative='two-sided')

            # statis, pvalue = stats.ttest_ind(a, b, equal_var=False)
            # s.append(u)
            # p.append(pvalue)
            p_result[i][3 * index + slice_ori] = pvalue

np.savetxt(os.path.join(fdir, 'stats', 'p_result.csv'), p_result)

if __name__ == '__main__':
    pass
    # f = np.load('feature/volume_feature/tumor_proportion_feature.npy')
    # p=[]
    # for i in range(f.shape[1]):
    #     # group a: meta
    #     a = f[0:45, i]
    #     # group b : gbm
    #     b = f[45:88, i]
    #     # plot data feature distribution between groups
    #     data_a = pd.DataFrame(a, columns=['meta'])
    #     data_b = pd.DataFrame(b, columns=['gbm'])
    #     plt.boxplot([data_a, data_b])
    #     plt.show()
    #     u, pvalue = stats.mannwhitneyu(a, b, alternative='two-sided')
    #     p.append(pvalue)
