import os
import time
import SimpleITK as sitk

from preprocess import contour

start = time.time()
## 从DWI中根据已有的label mask出tumor
## 仅在tumor区域计算GLCM的特征

inputdir1 = 'E:\data\ShandongHospitalBrain_preprocess_9\Brain_metastases_Solitary_Signa_3.0T'
inputdir2 = 'E:\data\ShandongHospitalBrain_preprocess_9\GBM_Signa_3.0T'
# inputdir1 = 'E:\data\BraTS2019\MICCAI_BraTS_2019_Data_Training\HGG'
# inputdir2 = 'E:\data\BraTS2019\MICCAI_BraTS_2019_Data_Training\LGG'
outputdir1 = 'E:\data\ShandongHospitalBrain_preprocess_9\Metastases_contour'
outputdir2 = 'E:\data\ShandongHospitalBrain_preprocess_9\GBM_contour'
# outputdir1 = 'E:\data\BraTS2019\MICCAI_BraTS_2019_Data_Training\HGG_Tumor'
# outputdir2 = 'E:\data\BraTS2019\MICCAI_BraTS_2019_Data_Training\LGG_Tumor'

patient_name_list1 = os.listdir(inputdir1)
patient_name_list2 = os.listdir(inputdir2)

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


for patient_name in patient_name_list1:
    filedir1 = inputdir1 + '/' + patient_name
    filelist1 = os.listdir(filedir1)
    print(filelist1)

    dwi = filelist1[4]  ##取 DWI 数据
    print(dwi)
    label = filelist1[1]  ##取 label 数据
    # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5

    dwiImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + dwi)
    labelImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + label)
    dwiNp = sitk.GetArrayFromImage(dwiImage)

    labelNp = sitk.GetArrayFromImage(labelImage)
    labelNp = contour(labelNp)

    dwiNp[labelNp == 0] = 0  #tumor
    # dwiNp[labelNp != 0] = 0  #normal

    dwiImage1 = sitk.GetImageFromArray(dwiNp)
    dwiImage1.CopyInformation(dwiImage)
    # if not os.path.exists(outputdir1 + '/' + patient_name):
    #     os.mkdir(outputdir1 + '/' + patient_name)
    sitk.WriteImage(dwiImage1, outputdir1 + '/' + patient_name + '/' +'T2-FLAIR.nii.gz')
#

# ----------------------------------------------------
for patient_name in patient_name_list2:
    filedir2 = inputdir2 + '/' + patient_name
    filelist2 = os.listdir(filedir2)
    print(filelist2)

    dwi = filelist2[4]  ##取 DWI 数据
    print(dwi)
    label = filelist2[1]  ##取 label 数据
    # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5

    dwiImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + dwi)
    labelImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + label)
    dwiNp = sitk.GetArrayFromImage(dwiImage)
    labelNp = sitk.GetArrayFromImage(labelImage)

    labelNp = sitk.GetArrayFromImage(labelImage)
    labelNp = contour(labelNp)

    dwiNp[labelNp == 0] = 0  #tumor
    # dwiNp[labelNp != 0] = 0  #normal

    dwiImage1 = sitk.GetImageFromArray(dwiNp)
    dwiImage1.CopyInformation(dwiImage)
    # if not os.path.exists(outputdir2 + '/' + patient_name):
    #     os.mkdir(outputdir2 + '/' + patient_name)
    sitk.WriteImage(dwiImage1, outputdir2 + '/' + patient_name + '/' +'T2-FLAIR.nii.gz')




end = time.time()

print(end - start)
