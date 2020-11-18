import os

import numpy as np


def load_feature():
    str = 'v6'
    X1 = np.load('feature/' + str + '/X_DWI_transverse.npy')
    X2 = np.load('feature/' + str + '/X_DWI_sagittal.npy')
    X3 = np.load('feature/' + str + '/X_DWI_coronal.npy')
    X4 = np.load('feature/' + str + '/X_T1+c_transverse.npy')
    X5 = np.load('feature/' + str + '/X_T1+c_sagittal.npy')
    X6 = np.load('feature/' + str + '/X_T1+c_coronal.npy')
    X7 = np.load('feature/' + str + '/X_T2_transverse.npy')
    X8 = np.load('feature/' + str + '/X_T2_sagittal.npy')
    X9 = np.load('feature/' + str + '/X_T2_coronal.npy')
    X10 = np.load('feature/' + str + '/X_T2-FLAIR_transverse.npy')
    X11 = np.load('feature/' + str + '/X_T2-FLAIR_sagittal.npy')
    X12 = np.load('feature/' + str + '/X_T2-FLAIR_coronal.npy')
    tumor_proportion = np.load('../feature/volume_feature/tumor_proportion_feature.npy')

    sexage1 = np.load('../meta_sex_age.npy')
    sexage2 = np.load('../gbm_sex_age.npy')
    sexage = np.vstack((sexage1, sexage2))
    name = sexage[:, 0]
    name = name.reshape((88, 1))
    sexage = sexage[:, 1:]
    sex = sexage[:, 0]
    sex = sex.reshape((88, 1))
    sexage = sexage.astype(np.int)
    sexage[:, 1] = sexage[:, 1] // 10

    namesex = np.hstack((name, sex))

    X = np.hstack((X1, X2, X3, X4, X5, X6, X7, X8, X9,
                   X10, X11, X12, tumor_proportion))
    y = np.load('../feature/v6/y.npy')
    y = y.ravel()

    return X, y, namesex


def new_load_feature():
    indir = '/home/cjy/data/10.18/match/feature'

    X1 = np.load(os.path.join(indir, 'DWI_transverse.npy'))
    X2 = np.load(os.path.join(indir, 'DWI_sagittal.npy'))
    X3 = np.load(os.path.join(indir, 'DWI_coronal.npy'))
    X4 = np.load(os.path.join(indir, 'T1+c_transverse.npy'))
    X5 = np.load(os.path.join(indir, 'T1+c_sagittal.npy'))
    X6 = np.load(os.path.join(indir, 'T1+c_coronal.npy'))
    X7 = np.load(os.path.join(indir, 'T2_transverse.npy'))
    X8 = np.load(os.path.join(indir, 'T2_sagittal.npy'))
    X9 = np.load(os.path.join(indir, 'T2_coronal.npy'))
    X10 = np.load(os.path.join(indir, 'T2-FLAIR_transverse.npy'))
    X11 = np.load(os.path.join(indir, 'T2-FLAIR_sagittal.npy'))
    X12 = np.load(os.path.join(indir, 'T2-FLAIR_coronal.npy'))
    # X13 = np.load(os.path.join(indir, 'label_transverse.npy'))
    # X14 = np.load(os.path.join(indir, 'label_sagittal.npy'))
    # X15 = np.load(os.path.join(indir, 'label_coronal.npy'))
    # tumor_proportion = np.load('feature/volume_feature/tumor_proportion_feature.npy')

    sexage1 = np.load('../meta_sex_age.npy')
    sexage2 = np.load('../gbm_sex_age.npy')
    sexage = np.vstack((sexage1, sexage2))
    name = sexage[:, 0]
    name = name.reshape((88, 1))
    sexage = sexage[:, 1:]
    sex = sexage[:, 0]
    sex = sex.reshape((88, 1))
    sexage = sexage.astype(np.int)
    sexage[:, 1] = sexage[:, 1] // 10

    namesex = np.hstack((name, sex))

    # all
    # X = np.hstack((X1, X2, X3, X4, X5, X6, X7, X8, X9,
    #                X10, X11, X12))
    # X = np.hstack((X1, X2, X3, X4, X5, X6, X7, X8, X9,
    #                X10, X11, X12, sexage))

    # transverse
    # X = np.hstack((X1, X4, X7, X10))
    # X = np.hstack((X2, X5, X8, X11))
    # X = np.hstack((X3, X6, X9, X12))

    # DWI
    # X = np.hstack((X1, X2, X3))
    # X = np.hstack((X4, X5, X6))
    X = np.hstack((X7, X8, X9))
    # X = np.hstack((X10, X11, X12))
    # X = np.hstack((X13, X14, X15))

    y = np.load(os.path.join(indir, 'y.npy'))
    y = y.ravel()

    return X, y, namesex


if __name__ == '__main__':
    sexage1 = np.load('../meta_sex_age.npy')
    sexage2 = np.load('../gbm_sex_age.npy')
