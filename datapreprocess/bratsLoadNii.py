import os
import time
import numpy as np
import SimpleITK as sitk
from feature import glcm
from preprocess import crop

start = time.time()

## 特征维数
HGG = np.ndarray(1 * 2640)
LGG = np.ndarray(1 * 2640)

inputdir1 = 'E:\data\BraTS2019\MICCAI_BraTS_2019_Data_Training\HGG_Tumor'
inputdir2 = 'E:\data\BraTS2019\MICCAI_BraTS_2019_Data_Training\LGG_Tumor'
patient_name_list1 = os.listdir(inputdir1)
patient_name_list2 = os.listdir(inputdir2)

for patient_name in patient_name_list1:
    filedir1 = inputdir1 + '/' + patient_name
    filelist1 = os.listdir(filedir1)
    # print(filelist1)
    # FLAIR:0      T1: 1    T1ce:2   T2:3
    file = filelist1[2]
    print(file)
    sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
    sitkNp = sitk.GetArrayFromImage(sitkImage)

    # 裁剪到一样大小
    sitkNp = crop(sitkNp, 123, 220, 220)
    sitkNp_int = sitkNp.astype(np.uint8)

    # ----------test 2D glcm
    # feature dummyhead
    feature = np.zeros((1, 4))
    for i in range(sitkNp_int.shape[1]):
        image = sitkNp_int[:, i, :]
        temp = glcm(image)
        feature = np.hstack((feature, temp))
    # 去除dummyhead
    feature = feature[:, 4:]
    # 每个patient堆叠
    HGG = np.vstack((HGG, feature))

# 去除dummyhead
HGG = HGG[1:, :]
# 产生对应的label  HGG：0
HGG_label = np.zeros((HGG.shape[0], 1))

# #---------------------------------------------
## 备注参考上面处理转移瘤的过程
for patient_name in patient_name_list2:
    filedir2 = inputdir2 + '/' + patient_name
    filelist2 = os.listdir(filedir2)
    # print(filelist2)
    file = filelist2[2]
    # FLAIR:0      T1: 1    T1ce:2   T2:3
    print(file)

    sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
    sitkNp = sitk.GetArrayFromImage(sitkImage)
    sitkNp = crop(sitkNp, 123, 220, 220)
    ## z x y
    sitkNp_int = sitkNp.astype(np.uint8)
    feature = np.zeros((1, 4))
    for i in range(sitkNp_int.shape[1]):
        image = sitkNp_int[:, i, :]
        # transverse  sagittal  coronal
        temp = glcm(image)
        feature = np.hstack((feature, temp))
    feature = feature[:, 4:]
    LGG = np.vstack((LGG, feature))
LGG = LGG[1:, :]
# 产生对应的label  LGG：1
LGG_label = np.ones((LGG.shape[0], 1))

# 整合数据
X = np.vstack((HGG, LGG))
y = np.vstack((HGG_label, LGG_label))

np.save('../feature/BraTS/X_T1ce_sagittal.npy', X)
np.save('../feature/BraTS/y.npy', y)

end = time.time()

print(end - start)