import os
import time

import SimpleITK as sitk
import numpy as np

from feature import glcm
from preprocess import crop, grayCompression

start = time.time()

# 特征维数
meta = np.ndarray((1, 16))
gbm = np.ndarray((1, 16))

inputdir1 = '/home/cjy/data/data_final/meta_tumor'
inputdir2 = '/home/cjy/data/data_final/GBM_tumor'

# inputdir1 = '/home/cjy/data/data_final/meta'
# inputdir2 = '/home/cjy/data/data_final/GBM'

patient_name_list1 = os.listdir(inputdir1)
patient_name_list1 = sorted(patient_name_list1)
patient_name_list2 = os.listdir(inputdir2)
patient_name_list2 = sorted(patient_name_list2)

index = 3
# DWI: 0   T1: 1   T2: 2    T2-FLAIR: 3
slice_ori = 0
# transverse : 0   sagittal: 1   coronal: 2

for patient_name in patient_name_list1:
    filedir1 = inputdir1 + '/' + patient_name
    filelist1 = os.listdir(filedir1)
    filelist1 = sorted(filelist1)
    # print(filelist1)
    file = filelist1[index]
    ##取数据
    print(file)

    sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
    sitkNp = sitk.GetArrayFromImage(sitkImage)
    # 裁剪到一样大小
    sitkNp = crop(sitkNp, 123, 220, 220)
    sitkNp = grayCompression(sitkNp)
    sitkNp_int = sitkNp.astype(np.uint8)

    # ----------test 2D glcm
    # feature dummyhead
    feature = np.zeros((1, 16))
    for i in range(sitkNp_int.shape[slice_ori]):
        if slice_ori == 0:
            image = sitkNp_int[i, :, :]
        elif slice_ori == 1:
            image = sitkNp_int[:, i, :]
        elif slice_ori == 2:
            image = sitkNp_int[:, :, i]
        else:
            print("slice_ori error ")
            break
        # print(image)
        if (image == np.zeros(image.shape)).all():
            continue
        temp = glcm(image)
        feature = np.vstack((feature, temp))
    # 去除dummyhead
    feature = feature[1:, :]
    mean_feature = np.mean(feature, axis=0)
    # 每个patient堆叠
    meta = np.vstack((meta, mean_feature))

# 去除dummyhead
meta = meta[1:, :]
# 产生对应的label  转移瘤：0
meta_label = np.zeros((meta.shape[0], 1))

# #---------------------------------------------
## 备注参考上面处理转移瘤的过程
for patient_name in patient_name_list2:
    filedir2 = inputdir2 + '/' + patient_name
    filelist2 = os.listdir(filedir2)
    filelist2 = sorted(filelist2)
    file = filelist2[index]
    print(file)

    sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
    sitkNp = sitk.GetArrayFromImage(sitkImage)
    sitkNp = crop(sitkNp, 123, 220, 220)
    sitkNp = grayCompression(sitkNp)
    sitkNp_int = sitkNp.astype(np.uint8)
    # ----------test 2D glcm
    # feature dummyhead
    feature = np.zeros((1, 16))

    for i in range(sitkNp_int.shape[slice_ori]):
        if slice_ori == 0:
            image = sitkNp_int[i, :, :]
        elif slice_ori == 1:
            image = sitkNp_int[:, i, :]
        elif slice_ori == 2:
            image = sitkNp_int[:, :, i]
        else:
            print("slice_ori error ")
            break
        # transverse  sagittal  coronal
        if (image == np.zeros(image.shape)).all():
            continue
        # print(np.max(image))
        temp = glcm(image)
        feature = np.vstack((feature, temp))
    # 去除dummyhead
    feature = feature[1:, :]
    gbm_feature = np.mean(feature, axis=0)

    # 每个patient堆叠
    gbm = np.vstack((gbm, gbm_feature))

# 去除dummyhead
gbm = gbm[1:, :]
# 产生对应的label  胶质瘤：1
gbm_label = np.ones((gbm.shape[0], 1))

# 整合数据
X = np.vstack((meta, gbm))
y = np.vstack((meta_label, gbm_label))

# DWI: 0   T1: 1   T2: 2    T2-FLAIR: 3
# transverse : 0   sagittal: 1   coronal: 2

mod = ['DWI', 'T1+c', 'T2', 'T2-FLAIR']
# mod = ['label', 'DWI', 'T1+c', 'T2', 'T2-FLAIR']
slice_oris = ['transverse', 'sagittal', 'coronal']

print(mod[index])
print(slice_oris[slice_ori])

np.save('feature/v5/X_' + mod[index] + '_' + slice_oris[slice_ori] + '.npy', X)
np.save('feature/v5/y.npy', y)

end = time.time()

print(end - start)
