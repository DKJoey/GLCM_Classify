import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

files = ['DWI_transverse.npy',
         'DWI_sagittal.npy',
         'DWI_coronal.npy',
         'T1+c_transverse.npy',
         'T1+c_sagittal.npy',
         'T1+c_coronal.npy',
         'T2_transverse.npy',
         'T2_sagittal.npy',
         'T2_coronal.npy',
         'T2-FLAIR_transverse.npy',
         'T2-FLAIR_sagittal.npy',
         'T2-FLAIR_coronal.npy']

for file in files:
    print(f'start processing', file)
    X = np.load(os.path.join('C:/Users\Administrator\Desktop/fl_match_to_first/tumor/feature', file))
    outdir = 'C:/Users\Administrator\Desktop/fl_match_to_first/tumor/feature/f_plot'

    X = preprocessing.scale(X)

    for i in range(X.shape[0]):
        print(i)
        # os.makedirs(os.path.join(outdir,str(i)))
        for j in range(0, X.shape[1], 4):
            f = X[i, j:j + 4]
            f = np.reshape(f, (1, 4))
            if j == 0:
                plt.imshow(f, cmap='jet')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(outdir, str(i), file[:-4] + '_Contrast.jpg'))
                plt.show()
            elif j == 4:
                plt.imshow(f, cmap='jet')
                plt.xticks([])
                plt.yticks([])
                # # plt.pcolormesh(f)
                plt.savefig(os.path.join(outdir, str(i), file[:-4] + '_Correlation.jpg'))
                plt.show()
            elif j == 8:
                plt.imshow(f, cmap='jet')
                plt.xticks([])
                plt.yticks([])
                # # plt.pcolormesh(f)
                plt.savefig(os.path.join(outdir, str(i), file[:-4] + '_Energy.jpg'))
                plt.show()
            elif j == 12:
                plt.imshow(f, cmap='jet')
                plt.xticks([])
                plt.yticks([])
                # # plt.pcolormesh(f)
                plt.savefig(os.path.join(outdir, str(i), file[:-4] + '_Homogeneity.jpg'))
                plt.show()
