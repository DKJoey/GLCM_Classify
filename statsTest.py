import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# feature statistic analysis

X = np.load('feature/v5nc/X_DWI_transverse.npy')
y = np.load('feature/v5/y.npy')

# X = preprocessing.scale(X)

X_m = np.ndarray((88, 4))

for i in range(88):
    for j in [0, 4, 8, 12]:
        X_m[i, j // 4] = np.mean(X[i, j:j + 3])

print(X_m)

result = np.ndarray((2, 8))

for i in [0, 2, 4, 6]:
    result[0, i] = np.mean(X_m[0:45, i // 2])
    result[0, i + 1] = np.var(X_m[0:45, i // 2])
    result[1, i] = np.mean(X_m[45:88, i // 2])
    result[1, i + 1] = np.var(X_m[45:88, i // 2])

print(result)

s = []
p = []
# v = []

for i in range(4):
    a = X_m[0:45, i]
    b = X_m[45:88, i]
    plt.boxplot([a, b])
    # plt.savefig('plot/v3/DWI_transverse_' + str(i) + '.jpg')
    plt.show()
    # statis, pvalue = stats.mannwhitneyu(a, b)
    # varis = stats.levene(a, b)
    # v.append(varis)
    statis, pvalue = stats.ttest_ind(a, b, equal_var='False')
    s.append(statis)
    p.append(pvalue)