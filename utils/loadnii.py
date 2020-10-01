import os
import time

import SimpleITK as sitk
import numpy as np

from utils.feature import glcm
from utils.preprocess import crop, grayCompression

start = time.time()

## 特征维数
meta = np.ndarray(1 * 2640)
gbm = np.ndarray(1 * 2640)

inputdir1 = 'E:\data\ShandongHospitalBrain_preprocess_7\Metastases_Tumor'
inputdir2 = 'E:\data\ShandongHospitalBrain_preprocess_7\GBM_Tumor'
patient_name_list1 = os.listdir(inputdir1)
patient_name_list2 = os.listdir(inputdir2)

for patient_name in patient_name_list1:
    filedir1 = inputdir1 + '/' + patient_name
    filelist1 = os.listdir(filedir1)
    print(filelist1)

    file = filelist1[0]  ##取 DWI 数据
    print(file)
    # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5
    #ROI
    # # DWI: 0  T1+c: 1   T2-FLAIR: 2   T2: 3

    sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
    sitkNp = sitk.GetArrayFromImage(sitkImage)
    # 裁剪到一样大小
    sitkNp = crop(sitkNp, 123, 220, 220)
    sitkNp = grayCompression(sitkNp)
    sitkNp_int = sitkNp.astype(np.uint8)

    # ----------test 2D glcm
    # feature dummyhead
    feature = np.zeros((1, 4))
    for i in range(sitkNp_int.shape[0]):
        image = sitkNp_int[i, :, :]
        temp = glcm(image)
        feature = np.hstack((feature, temp))
    # 去除dummyhead
    feature = feature[:, 4:]
    # 每个patient堆叠
    meta = np.vstack((meta, feature))

# 去除dummyhead
meta = meta[1:, :]
# 产生对应的label  转移瘤：0
meta_label = np.zeros((meta.shape[0], 1))

# #---------------------------------------------
## 备注参考上面处理转移瘤的过程
for patient_name in patient_name_list2:
    filedir2 = inputdir2 + '/' + patient_name
    filelist2 = os.listdir(filedir2)
    print(filelist2)
    file = filelist2[0]
    print(file)
    # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5
    sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
    sitkNp = sitk.GetArrayFromImage(sitkImage)
    sitkNp = crop(sitkNp, 123, 220, 220)
    sitkNp = grayCompression(sitkNp)
    ## z x y
    sitkNp_int = sitkNp.astype(np.uint8)
    feature = np.zeros((1, 4))
    for i in range(sitkNp_int.shape[0]):
        image = sitkNp_int[i, :, :]
        # transverse  sagittal  coronal
        temp = glcm(image)
        feature = np.hstack((feature, temp))
    feature = feature[:, 4:]
    gbm = np.vstack((gbm, feature))
gbm = gbm[1:, :]
# 产生对应的label  胶质瘤：1
gbm_label = np.ones((gbm.shape[0], 1))

# 整合数据
X = np.vstack((meta, gbm))
y = np.vstack((meta_label, gbm_label))

np.save('feature/v3/X_DWI_transverse.npy', X)
np.save('feature/v3/y.npy', y)

end = time.time()

print(end - start)