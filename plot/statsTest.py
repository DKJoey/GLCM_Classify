import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if __name__ == '__main__':
    # feature statistic analysis

    rootdir = '/home/cjy/data/10.11'
    fold = 'match'
    featuredir = 'feature'

    fdir = os.path.join(rootdir, fold, featuredir)

    mod = ['DWI', 'T1+c', 'T2', 'T2-FLAIR']
    slice_oris = ['transverse', 'sagittal', 'coronal']

    p_result = np.zeros((16, 12))

    for index in range(4):
        for slice_ori in range(3):
            X = np.load(os.path.join(fdir, mod[index] + '_' + slice_oris[slice_ori] + '.npy'))

            for i in range(X.shape[1]):
                # group a: meta
                a = X[0:45, i]
                # group b : gbm
                b = X[45:88, i]
                # plot data feature distribution between groups
                data_a = pd.DataFrame(a, columns=['meta'])
                data_b = pd.DataFrame(b, columns=['gbm'])
                plt.boxplot([data_a, data_b])

                plt.savefig(
                    os.path.join(fdir, 'stats', mod[index] + '_' + slice_oris[slice_ori] + '_' + str(i) + '.jpg'))
                plt.show()

                u, pvalue = stats.mannwhitneyu(a, b, alternative='two-sided')
                p_result[i][3 * index + slice_ori] = pvalue

    np.savetxt(os.path.join(fdir, 'stats', 'p_result.csv'), p_result)
    np.save(os.path.join(fdir, 'stats', 'p_result.npy'), p_result)

    # ---------------------------------------
    # roc result t test
    # a = [84.73, 91.52, 93.23, 72.92, 82.58, 71.14, 87.08]
    # b = [86.64, 93.64, 83.28, 68.22, 85.22, 68.48, 89.15]
    #
    # t, p = stats.ttest_rel(a, b)
