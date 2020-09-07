import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# feature statistic analysis

mod = ['DWI', 'T1+c', 'T2', 'T2-FLAIR']
slice_oris = ['transverse', 'sagittal', 'coronal']

for index in range(4):
    for slice_ori in range(3):
        X = np.load('feature/v5whole/X_' + mod[index] + '_' + slice_oris[slice_ori] + '.npy')

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

            plt.savefig('plot/v5whole/' + mod[index] + '_' + slice_oris[slice_ori] + '_' + str(i) + '.jpg')
            plt.show()

            u, pvalue = stats.mannwhitneyu(a, b, alternative='two-sided')

            # statis, pvalue = stats.ttest_ind(a, b, equal_var=False)
            s.append(u)
            p.append(pvalue)
        np.savetxt('plot/v5whole/' + mod[index] + '_' + slice_oris[slice_ori] + '.txt', (s, p))
