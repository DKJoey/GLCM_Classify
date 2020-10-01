import os
import time

import SimpleITK as sitk

start = time.time()
## 从DWI中根据已有的label mask出tumor
## 仅在tumor区域计算GLCM的特征

inputdir1 = '/home/cjy/data/comp_pre/fl_match_to_first/meta'
inputdir2 = '/home/cjy/data/comp_pre/fl_match_to_first/GBM'

outputdir1 = '/home/cjy/data/comp_pre/fl_match_to_first/tumor/meta'
outputdir2 = '/home/cjy/data/comp_pre/fl_match_to_first/tumor/GBM'

patient_name_list1 = sorted(os.listdir(inputdir1))
patient_name_list2 = sorted(os.listdir(inputdir2))

##---BraTS
# for patient_name in patient_name_list2:
#     filedir2 = inputdir2 + '/' + patient_name
#     filelist2 = os.listdir(filedir2)
#     # print(filelist2)
#     # FLAIR:0    Seg:1    T1: 2    T1ce:3   T2:4
#     flair = filelist2[0]
#     print(flair)
#     label = filelist2[1]
#     flairImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + flair)
#     labelImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + label)
#     flairNp = sitk.GetArrayFromImage(flairImage)
#     labelNp = sitk.GetArrayFromImage(labelImage)
#
#     flairNp[labelNp == 0] = 0
#
#     flairImage1 = sitk.GetImageFromArray(flairNp)
#     flairImage1.CopyInformation(flairImage)
#     if not os.path.exists(outputdir2 + '/' + patient_name):
#         os.mkdir(outputdir2 + '/' + patient_name)
#     sitk.WriteImage(flairImage1, outputdir2 + '/' + patient_name + '/' + 'FLAIR.nii.gz')

index = 3

for patient_name in patient_name_list1:
    filedir1 = inputdir1 + '/' + patient_name
    filelist1 = sorted(os.listdir(filedir1))
    # print(filelist1)

    dwi = filelist1[index]  ##取 DWI 数据
    print(dwi)
    label = filelist1[0]  ##取 label 数据
    #    Label: 0       T1+c: 1  DWI :2    T2: 3   T2-FLAIR: 4

    dwiImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + dwi)
    labelImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + label)
    dwiNp = sitk.GetArrayFromImage(dwiImage)
    labelNp = sitk.GetArrayFromImage(labelImage)
    # labelNp = contour(labelNp)

    # dwiNp[dwiNp < 0] = 0

    dwiNp[labelNp == 0] = 0  # tumor
    # dwiNp[labelNp != 0] = 0  #normal

    dwiImage1 = sitk.GetImageFromArray(dwiNp)
    dwiImage1.CopyInformation(dwiImage)
    if not os.path.exists(outputdir1 + '/' + patient_name):
        os.mkdir(outputdir1 + '/' + patient_name)
    sitk.WriteImage(dwiImage1, outputdir1 + '/' + patient_name + '/' + dwi)
#

'''GBM'''
for patient_name in patient_name_list2:
    filedir2 = inputdir2 + '/' + patient_name
    filelist2 = sorted(os.listdir(filedir2))
    dwi = filelist2[index]  ##取 DWI 数据
    print(dwi)
    label = filelist2[0]  ##取 label 数据
    #    Label: 0       T1+c: 2  DWI :1    T2: 3   T2-FLAIR: 4

    dwiImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + dwi)
    labelImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + label)
    dwiNp = sitk.GetArrayFromImage(dwiImage)
    labelNp = sitk.GetArrayFromImage(labelImage)
    # labelNp = contour(labelNp)

    # dwiNp[dwiNp < 0] = 0

    dwiNp[labelNp == 0] = 0  # tumor
    # dwiNp[labelNp != 0] = 0  #normal

    dwiImage1 = sitk.GetImageFromArray(dwiNp)
    dwiImage1.CopyInformation(dwiImage)
    if not os.path.exists(outputdir2 + '/' + patient_name):
        os.mkdir(outputdir2 + '/' + patient_name)
    sitk.WriteImage(dwiImage1, outputdir2 + '/' + patient_name + '/' + dwi)


end = time.time()

print(end - start)
