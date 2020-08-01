import numpy as np


def load_feature():
    X1 = np.load('feature/v5/X_DWI_transverse.npy')
    X2 = np.load('feature/v5/X_DWI_sagittal.npy')
    X3 = np.load('feature/v5/X_DWI_coronal.npy')
    X4 = np.load('feature/v5/X_T1+c_transverse.npy')
    X5 = np.load('feature/v5/X_T1+c_sagittal.npy')
    X6 = np.load('feature/v5/X_T1+c_coronal.npy')
    X7 = np.load('feature/v5/X_T2_transverse.npy')
    X8 = np.load('feature/v5/X_T2_sagittal.npy')
    X9 = np.load('feature/v5/X_T2_coronal.npy')
    X10 = np.load('feature/v5/X_T2-FLAIR_transverse.npy')
    X11 = np.load('feature/v5/X_T2-FLAIR_sagittal.npy')
    X12 = np.load('feature/v5/X_T2-FLAIR_coronal.npy')

    sexage1 = np.load('meta_sex_age.npy')
    sexage2 = np.load('gbm_sex_age.npy')
    sexage = np.vstack((sexage1, sexage2))
    name = sexage[:, 0]
    name = name.reshape((88, 1))
    sexage = sexage[:, 1:]
    sex = sexage[:, 0]
    sex = sex.reshape((88, 1))
    sexage = sexage.astype(np.int)
    sexage[:, 1] = sexage[:, 1] // 10

    namesex = np.hstack((name, sex))

    X = np.hstack((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12))
    y = np.load('feature/v5/y.npy')
    y = y.ravel()

    return X, y, namesex
