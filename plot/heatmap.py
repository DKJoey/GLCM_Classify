import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.load_feature import new_load_feature

# feature heatmap
if __name__ == "__main__":
    sexage1 = np.load('../meta_sex_age.npy')
    sexage2 = np.load('../gbm_sex_age.npy')
    sexage1 = sexage1[:, 1:].astype(np.int)
    sexage2 = sexage2[:, 1:].astype(np.int)

    sexage1_male = np.sum(sexage1[:, 0])
    sexage2_male = np.sum(sexage2[:, 0])

    sexage1_age_avg = np.mean(sexage1[:, 1])
    sexage2_age_avg = np.mean(sexage2[:, 1])
    sexage1_age_std = np.std(sexage1[:, 1])
    sexage2_age_std = np.std(sexage2[:, 1])

    np.savetxt('../meta_sex_age.csv', sexage1, fmt='%s')
    np.savetxt('../gbm_sex_age.csv', sexage2, fmt='%s')

    sexage = np.vstack((sexage1, sexage2))
    sexage = sexage[:, 1:]
    sexage = sexage.astype(np.int)

    X, y, _ = new_load_feature()
    y = y.reshape((88, 1))
    odir = '/home/cjy/data/10.11/match/plot'
    data = X
    #
    # data = np.hstack((y, sexage))
    # print(data.shape)
    # label = ['Class', 'Sex', 'Age']
    pdata = pd.DataFrame(data)

    # file = np.load('/home/cjy/data/10.11/match/feature/stats/p_result.npy')
    # corrmat = pd.DataFrame(file)
    # f, ax = plt.subplots()
    # sns.heatmap(corrmat.iloc[:, :], square=True, cmap="rainbow", xticklabels=True,
    #             yticklabels=True)

    corrmat = pd.DataFrame(data).corr()
    corrmat = abs(corrmat)
    f, ax = plt.subplots()
    sns.heatmap(corrmat.iloc[:, :], square=True, cmap="rainbow", vmax=1, vmin=0, xticklabels=False,
                yticklabels=False)

    plt.savefig(os.path.join(odir, 'heatmap'))
    plt.show()
