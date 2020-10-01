import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.manifold import TSNE

# TSNE for visualization

inputdir1 = 'E:\data\ShandongHospitalBrain_preprocess_9\Metastases_Tumor'
inputdir2 = 'E:\data\ShandongHospitalBrain_preprocess_9\GBM_Tumor'
patient_name_list1 = os.listdir(inputdir1)
patient_name_list2 = os.listdir(inputdir2)
patient_name_list1 = np.array(patient_name_list1)
patient_name_list1 = patient_name_list1.reshape((45, 1))
patient_name_list2 = np.array(patient_name_list2)
patient_name_list2 = patient_name_list2.reshape((43, 1))
patient_name_list = np.vstack((patient_name_list1, patient_name_list2))

X1 = np.load('../feature/v5/X_DWI_transverse.npy')
X2 = np.load('../feature/v5/X_DWI_sagittal.npy')
X3 = np.load('../feature/v5/X_DWI_coronal.npy')
X4 = np.load('../feature/v5/X_T1+c_transverse.npy')
X5 = np.load('../feature/v5/X_T1+c_sagittal.npy')
X6 = np.load('../feature/v5/X_T1+c_coronal.npy')
X7 = np.load('../feature/v5/X_T2_transverse.npy')
X8 = np.load('../feature/v5/X_T2_sagittal.npy')
X9 = np.load('../feature/v5/X_T2_coronal.npy')
X10 = np.load('../feature/v5/X_T2-FLAIR_transverse.npy')
X11 = np.load('../feature/v5/X_T2-FLAIR_sagittal.npy')
X12 = np.load('../feature/v5/X_T2-FLAIR_coronal.npy')
X = np.hstack((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12))
y = np.load('../feature/v5/y.npy')
y = y.ravel()

result = np.hstack((patient_name_list, X))

np.savetxt('feature.csv', result, delimiter=',', fmt='%s')

# tsne = TSNE()
# tsne.fit_transform(X)  # 进行数据降维,降成两维
# tsne = tsne.embedding_
# tsne1 = tsne[0:45]
# tsne2 = tsne[45:]
# plt.plot(tsne1, 'r.')
# plt.plot(tsne2, 'b*')
# plt.show()
#
# tsne = TSNE(n_components=3)
# tsne.fit_transform(X)  # 进行数据降维,降成两维
# tsne = tsne.embedding_
# tsne1 = tsne[0:45]
# tsne2 = tsne[45:]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # 将二维转化为三维
# axes3d = Axes3D(fig)
# # axes3d.scatter3D(x,y,z)
# # 效果相同
# axes3d.scatter(tsne1[:, 0], tsne1[:, 1], tsne1[:, 2], c='r')
# axes3d.scatter(tsne2[:, 0], tsne2[:, 1], tsne2[:, 2], c='b')
# plt.show()
