import glob
import os

import SimpleITK as sitk
import numpy as np


def listdir_with_postfix(path, postfix):
    return sorted(glob.glob(os.path.join(path, ('*'+postfix))))

inputdir1 = 'E:\data\sex_age\Brain_metastases_Solitary_Signa_3.0T'
inputdir2 = 'E:\data\sex_age\GBM_Signa_3.0T'
patient_name_list1 = os.listdir(inputdir1)
patient_name_list2 = os.listdir(inputdir2)

table1 = []
table2 = []

# for patient_name in patient_name_list1:
#     filedir1 = inputdir1 + '/' + patient_name + '/T2-FLAIR'
#     dicom_list = listdir_with_postfix(filedir1, '.dcm')
#     reader = sitk.ImageSeriesReader()
#     reader.MetaDataDictionaryArrayUpdateOn()  # 这一步是加载公开的元信息
#     reader.LoadPrivateTagsOn()  # 这一步是加载私有的元信息
#     reader.SetFileNames(dicom_list)  # 设置文件名
#     image3D = reader.Execute()  # 读取dicom序列
#     size=image3D.GetSize()
#     spacing=image3D.GetSpacing()
#
#     # for key in reader.GetMetaDataKeys(0):
#     #     value = reader.GetMetaData(0,key)
#     #     print("({0}) = = \"{1}\"".format(key, value))
#
#     sex =reader.GetMetaData(0,'0010|0040')
#     # M:1 F：0
#     if sex == 'M ':
#         sexnum= 1
#     else:
#         sexnum=0
#     age =reader.GetMetaData(0,'0010|1010')
#     # age:0**y
#     agenum = age[1:-1]
#
#     fieldStrength = reader.GetMetaData(0,'0018|0087')
#     #场强
#     brand = reader.GetMetaData(0,'0008|0070')
#     #厂商
#     modal = reader.GetMetaData(0,'0008|103e')
#     #模态
#
#     table1.append((patient_name,sex,age,fieldStrength,brand,modal,size,spacing))
# #
# print(table1)
# table1 = np.array(table1)
# print(table1)
# # np.save('meta_sex_age.npy',table1)
# np.savetxt('metadata/meta_T2-FLAIR.txt',table1,fmt='%s',delimiter=' ')


for patient_name in patient_name_list2:
    filedir2 = inputdir2 + '/' + patient_name + '/DWI'
    dicom_list = listdir_with_postfix(filedir2, '.dcm')
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()  # 这一步是加载公开的元信息
    reader.LoadPrivateTagsOn()  # 这一步是加载私有的元信息
    reader.SetFileNames(dicom_list)  # 设置文件名
    image3D = reader.Execute()  # 读取dicom序列

    size=image3D.GetSize()
    spacing=image3D.GetSpacing()

    # for key in reader.GetMetaDataKeys(0):
    #     value = reader.GetMetaData(0,key)
    #     print("({0}) = = \"{1}\"".format(key, value))

    sex =reader.GetMetaData(0,'0010|0040')
    # M:1 F：0
    if sex == 'M ':
        sexnum= 1
    else:
        sexnum=0
    age =reader.GetMetaData(0,'0010|1010')
    # age:0**y
    agenum = age[1:-1]

    fieldStrength = reader.GetMetaData(0,'0018|0087')
    #场强
    brand = reader.GetMetaData(0,'0008|0070')
    #厂商
    modal = reader.GetMetaData(0,'0008|103e')
    #模态

    table2.append((patient_name,sex,age,fieldStrength,brand,modal,size,spacing))
#
print(table2)
table2 = np.array(table2)
print(table2)
# np.save('gbm_sex_age.npy',table2)
np.savetxt('metadata/gbm_dwi-Axi.txt',table2,fmt='%s',delimiter=' ')
