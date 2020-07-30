import os

import matplotlib.pyplot as plt
import numpy as np

# data intensity histogram

inputdir1 = 'E:\data\ShandongHospitalBrain_preprocess_9\Brain_metastases_Solitary_Signa_3.0T'
inputdir2 = 'E:\data\ShandongHospitalBrain_preprocess_9\GBM_Signa_3.0T'

outputdir1 = 'E:\data\ShandongHospitalBrain_preprocess_10\Brain_metastases_Solitary_Signa_3.0T'
outputdir2 = 'E:\data\ShandongHospitalBrain_preprocess_10\GBM_Signa_3.0T'

patient_name_list1 = os.listdir(outputdir1)
patient_name_list2 = os.listdir(outputdir2)

# for patient_name in patient_name_list1:
#     filedir1 = inputdir1 + '/' + patient_name
#     filelist1 = os.listdir(filedir1)
#     print(filelist1)
#
#     file = filelist1[5]  ##取 DWI 数据
#     print(file)
#     # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5
#
#     sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
#     sitkNp = sitk.GetArrayFromImage(sitkImage)
#
#     # sitkNp = crop(sitkNp, 123, 220, 220)
#     sitkNp = grayCompression(sitkNp)
#     sitkNp_int = sitkNp.astype(np.uint8)
#
#     sitkImage1 = sitk.GetImageFromArray(sitkNp_int)
#
#     sitkImage1.CopyInformation(sitkImage)
#     # if not os.path.exists(outputdir1 + '/' + patient_name):
#     #     os.mkdir(outputdir1 + '/' + patient_name)
#     sitk.WriteImage(sitkImage1, outputdir1 + '/' + patient_name + '/' + file)
#
# for patient_name in patient_name_list2:
#     filedir2 = inputdir2 + '/' + patient_name
#     filelist2 = os.listdir(filedir2)
#     print(filelist2)
#
#     file = filelist2[5]  ##取 DWI 数据
#     print(file)
#     # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5
#
#     sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
#     sitkNp = sitk.GetArrayFromImage(sitkImage)
#
#     # sitkNp = crop(sitkNp, 123, 220, 220)
#     sitkNp = grayCompression(sitkNp)
#     sitkNp_int = sitkNp.astype(np.uint8)
#
#     sitkImage1 = sitk.GetImageFromArray(sitkNp_int)
#
#     sitkImage1.CopyInformation(sitkImage)
#     # if not os.path.exists(outputdir1 + '/' + patient_name):
#     #     os.mkdir(outputdir1 + '/' + patient_name)
#     sitk.WriteImage(sitkImage1, outputdir2 + '/' + patient_name + '/' + file)


if __name__ == '__main__':
    # count = np.zeros((45, 256))
    # patient_name_list1 = os.listdir(outputdir1)
    #
    # for index, patient_name in enumerate(patient_name_list1):
    #     filedir1 = outputdir1 + '/' + patient_name
    #     filelist1 = os.listdir(filedir1)
    #     print(filelist1)
    #
    #     file = filelist1[5]
    #     # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5
    #
    #     sitkImage = sitk.ReadImage(outputdir1 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #
    #     for i in range(sitkNp.shape[0]):
    #         for j in range(sitkNp.shape[1]):
    #             for k in range(sitkNp.shape[2]):
    #                 count[index, sitkNp[i, j, k]] += 1
    #
    # count1 = np.zeros((43, 256))
    # patient_name_list2 = os.listdir(outputdir2)
    #
    # for index, patient_name in enumerate(patient_name_list2):
    #     filedir2 = outputdir2 + '/' + patient_name
    #     filelist2 = os.listdir(filedir2)
    #     print(filelist2)
    #
    #     file = filelist2[5]
    #     # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5
    #
    #     sitkImage = sitk.ReadImage(outputdir2 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #
    #     for i in range(sitkNp.shape[0]):
    #         for j in range(sitkNp.shape[1]):
    #             for k in range(sitkNp.shape[2]):
    #                 count1[index, sitkNp[i, j, k]] += 1
    #
    # # np.save('histogram/T2-FLAIR', count)
    # np.save('histogram/T2-FLAIR1', count1)

    count = np.load('histogram/T2-FLAIR.npy')
    count1 = np.load('histogram/T2-FLAIR1.npy')
    count = np.vstack((count,count1))
    for i in range(88):
        plt.plot(count[i,1:])
    plt.show()