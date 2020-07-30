import os
import time
import numpy as np
import SimpleITK as sitk
from feature import glcm
from preprocess import crop, grayCompression

start = time.time()

## 特征维数
meta = np.ndarray(1 * 16)
gbm = np.ndarray(1 * 16)

inputdir1 = 'E:\data\ShandongHospitalBrain_preprocess_9\Metastases_contour'
inputdir2 = 'E:\data\ShandongHospitalBrain_preprocess_9\GBM_contour'
patient_name_list1 = os.listdir(inputdir1)
patient_name_list2 = os.listdir(inputdir2)

for patient_name in patient_name_list1:
    filedir1 = inputdir1 + '/' + patient_name
    filelist1 = os.listdir(filedir1)

    file = filelist1[2]  ##取 DWI 数据
    print(file)
    # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5
    # ROI
    ## DWI: 0  T1+c: 1   T2-FLAIR: 2   T2: 3

    sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
    sitkNp = sitk.GetArrayFromImage(sitkImage)
    # 裁剪到一样大小
    sitkNp = crop(sitkNp, 123, 220, 220)
    sitkNp = grayCompression(sitkNp)
    sitkNp_int = sitkNp.astype(np.uint8)

    # ----------test 2D glcm
    # feature dummyhead
    feature = np.zeros((1, 16))
    for i in range(sitkNp_int.shape[1]):
        image = sitkNp_int[:, i, :]
        # print(image)
        if (image == np.zeros(image.shape)).all():
            continue
        # print(np.max(image))
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
    file = filelist2[2]
    print(file)
    # # DWI: 0  T1+c: 1   T2-FLAIR: 2   T2: 3
    # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5
    sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
    sitkNp = sitk.GetArrayFromImage(sitkImage)
    sitkNp = crop(sitkNp, 123, 220, 220)
    sitkNp = grayCompression(sitkNp)
    ## z x y

    sitkNp_int = sitkNp.astype(np.uint8)
    # ----------test 2D glcm
    # feature dummyhead
    feature = np.zeros((1, 16))
    for i in range(sitkNp_int.shape[1]):
        image = sitkNp_int[:, i, :]
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

np.save('feature/v5/X_T2-FLAIR_sagittal.npy', X)
np.save('feature/v5/y.npy', y)

end = time.time()

print(end - start)
