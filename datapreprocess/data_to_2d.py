import os

import SimpleITK as sitk
import numpy as np
from PIL import Image

from preprocess import crop, grayCompression

inputdir1 = '/home/cjy/data/data_final/meta_tumor'
inputdir2 = '/home/cjy/data/data_final/GBM_tumor'

outdir = '/home/cjy/PycharmProjects/EffeientNet/data'

patient_name_list1 = os.listdir(inputdir1)
patient_name_list1 = sorted(patient_name_list1)
patient_name_list2 = os.listdir(inputdir2)
patient_name_list2 = sorted(patient_name_list2)

index = 3
# DWI: 0   T1: 1   T2: 2    T2-FLAIR: 3
slice_ori = 2
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

    for i in range(sitkNp_int.shape[slice_ori]):
        if slice_ori == 0:
            image = sitkNp_int[i, :, :]
        elif slice_ori == 1:
            image = sitkNp_int[:, i, :]
        elif slice_ori == 2:
            image = sitkNp_int[:, :, i]
        else:
            print("slice_ori error")
            break
        # print(image)
        if (image == np.zeros(image.shape)).all():
            continue
        im = Image.fromarray(image)
        im.show()
