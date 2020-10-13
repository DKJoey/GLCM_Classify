import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# data intensity histogram
from utils.preprocess import grayCompression

if __name__ == '__main__':
    count1 = np.zeros((45, 256))
    count2 = np.zeros((43, 256))
    #
    index = 1
    mod = ['label', 'DWI', 'T1+c', 'T2', 'T2-FLAIR', 'mask']

    rootdir = '/home/cjy/data/10.11'
    fold = 'match2'

    inputdir1 = os.path.join(rootdir, fold, 'meta')
    inputdir2 = os.path.join(rootdir, fold, 'GBM')

    patient_name_list1 = os.listdir(inputdir1)
    patient_name_list1 = sorted(patient_name_list1)
    patient_name_list2 = os.listdir(inputdir2)
    patient_name_list2 = sorted(patient_name_list2)

    for pi, patient_name in enumerate(patient_name_list1):
        filedir1 = inputdir1 + '/' + patient_name
        filelist1 = os.listdir(filedir1)
        filelist1 = sorted(filelist1)
        # print(filelist1)
        file = filelist1[index]
        # mask = filelist1[5]

        print(file)
        # print(mask)

        sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
        sitkNp = sitk.GetArrayFromImage(sitkImage)
        sitkNp = grayCompression(sitkNp)
        sitkNp = sitkNp.astype(np.uint8)

        # maskImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + mask)
        # maskNp = sitk.GetArrayFromImage(maskImage)

        for i in range(sitkNp.shape[0]):
            for j in range(sitkNp.shape[1]):
                for k in range(sitkNp.shape[2]):
                    # if maskNp[i, j, k] == 1:
                    count1[pi, sitkNp[i, j, k]] += 1

    for pi, patient_name in enumerate(patient_name_list2):
        filedir2 = inputdir2 + '/' + patient_name
        filelist2 = os.listdir(filedir2)
        filelist2 = sorted(filelist2)
        # print(filelist2)
        file = filelist2[index]
        # mask = filelist2[5]
        print(file)
        # print(mask)

        sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
        sitkNp = sitk.GetArrayFromImage(sitkImage)
        sitkNp = grayCompression(sitkNp)
        sitkNp = sitkNp.astype(np.uint8)

        # maskImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + mask)
        # maskNp = sitk.GetArrayFromImage(maskImage)

        for i in range(sitkNp.shape[0]):
            for j in range(sitkNp.shape[1]):
                for k in range(sitkNp.shape[2]):
                    # if maskNp[i, j, k] == 1:
                    count2[pi, sitkNp[i, j, k]] += 1

    # mod = ['label', 'DWI', 'T1+c', 'T2', 'T2-FLAIR']
    # np.save('histogram/matched_meta_' + mod[index] + '.npy', count1)
    # np.save('histogram/matched_GBM_' + mod[index] + '.npy', count2)

    # count1 = np.load('histogram/matched_meta_T2-FLAIR.npy')
    # count2 = np.load('histogram/matched_GBM_T2-FLAIR.npy')
    # count = np.vstack((count1, count2))

    plt.subplot(221)
    for i in range(45):
        plt.plot(count1[i, 1:])

    plt.subplot(222)
    for i in range(43):
        plt.plot(count2[i, 1:])

    plt.subplot(212)
    count = np.vstack((count1, count2))
    for i in range(88):
        plt.plot(count[i, 1:])

    plt.savefig(os.path.join(rootdir, fold, mod[index] + '.jpg'))
    plt.show()
