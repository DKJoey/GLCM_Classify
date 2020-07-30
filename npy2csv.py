import numpy as np

X_Dc = np.load('feature/X_DWI_coronal.npy')
X_Ds = np.load('feature/X_DWI_sagittal.npy')
X_Dt = np.load('feature/X_DWI_transverse.npy')

X_1c = np.load('feature/X_T1+c_coronal.npy')
X_1s = np.load('feature/X_T1+c_sagittal.npy')
X_1t = np.load('feature/X_T1+c_transverse.npy')

X_2c = np.load('feature/X_T2_coronal.npy')
X_2s = np.load('feature/X_T2_sagittal.npy')
X_2t = np.load('feature/X_T2_transverse.npy')

X_Fc = np.load('feature/X_T2-FLAIR_coronal.npy')
X_Fs = np.load('feature/X_T2-FLAIR_sagittal.npy')
X_Ft = np.load('feature/X_T2-FLAIR_transverse.npy')

np.savetxt('data/X_DWI_coronal.csv', X_Dc, delimiter=',')
