import numpy as np
import skimage.feature


# 灰度共生矩阵
def glcm(image):
    g = skimage.feature.greycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    # 对比度
    contrast = skimage.feature.greycoprops(g, prop='contrast')
    # 非相似性
    # dissimilarity = skimage.feature.greycoprops(g, prop='dissimilarity')
    # 同质度
    homogeneity = skimage.feature.greycoprops(g, prop='homogeneity')
    # 能量
    energy = skimage.feature.greycoprops(g, prop='energy')
    # 相关性
    correlation = skimage.feature.greycoprops(g, prop='correlation')
    # 角二阶矩
    # ASM = skimage.feature.greycoprops(g, prop='ASM')

    ans = np.hstack((contrast, correlation, energy, homogeneity))
    return ans


def cul_volume(image, value, isequal):
    ans = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                if isequal and image[i][j][k] == value:
                    ans += 1
                if not isequal and image[i][j][k] != value:
                    ans += 1
    return ans


def tumor_proportion_feature():
    meta = np.loadtxt('feature/volume_feature/volumecount_meta.csv')
    gbm = np.loadtxt('feature/volume_feature/volumecount_gbm.csv')
    ##'reinforce', 'nero', 'edema', 'brain1', 'brain2', 'brain3', 'brain4'
    f = np.vstack((meta, gbm))
    ans = np.zeros((88, 5))
    for i in range(88):
        brain_volume = np.mean(f[i, 3:])
        tumor_volume = f[i, 0] + f[i, 1] + f[i, 2]
        tumor_proportion = tumor_volume / brain_volume
        reinforce_proportion = f[i, 0] / tumor_volume
        nero_proportion = f[i, 1] / tumor_volume
        edema_proportion = f[i, 2] / tumor_volume
        ans[i, :] = np.asarray(
            [tumor_volume, tumor_proportion, reinforce_proportion, nero_proportion, edema_proportion])
    print(ans)
    np.save('feature/volume_feature/tumor_proportion_feature.npy', ans)


if __name__ == '__main__':
    # a= np.loadtxt('volumecount_gbm.csv')
    tumor_proportion_feature()
    pass
    # inputdir1 = '/home/cjy/data/data_final/meta_bg0'
    # inputdir2 = '/home/cjy/data/data_final/GBM_bg0'
    #
    # patient_name_list1 = os.listdir(inputdir1)
    # patient_name_list1 = sorted(patient_name_list1)
    # patient_name_list2 = os.listdir(inputdir2)
    # patient_name_list2 = sorted(patient_name_list2)
    #
    # # volumecount = ['reinforce', 'nero', 'edema', 'brain1', 'brain2', 'brain3', 'brain4']
    # volumecount = np.ndarray((45, 7))
    # for i, patient_name in enumerate(patient_name_list1):
    #     filedir1 = inputdir1 + '/' + patient_name
    #     filelist1 = os.listdir(filedir1)
    #     filelist1 = sorted(filelist1)
    #     print(filelist1)
    #     file = filelist1[0]
    #     ##取label
    #     print(file)
    #     sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #
    #     # reinforce
    #     volumecount[i][0] = cul_volume(sitkNp, 1, True)
    #     # nero
    #     volumecount[i][1] = cul_volume(sitkNp, 2, True)
    #     # edema
    #     volumecount[i][2] = cul_volume(sitkNp, 3, True) + cul_volume(sitkNp, 4, True) + cul_volume(sitkNp, 5, True)
    #
    #     file = filelist1[1]
    #     print(file)
    #     sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #     volumecount[i][3] = np.count_nonzero(sitkNp)
    #
    #     file = filelist1[2]
    #     print(file)
    #     sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #     volumecount[i][4] = np.count_nonzero(sitkNp)
    #
    #     file = filelist1[3]
    #     print(file)
    #     sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #     volumecount[i][5] = np.count_nonzero(sitkNp)
    #
    #     file = filelist1[4]
    #     print(file)
    #     sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #     volumecount[i][6] = np.count_nonzero(sitkNp)
    #
    # np.savetxt('feature/volume_feature/volumecount_meta.csv', volumecount)

    # volumecount = ['reinforce', 'nero', 'edema', 'brain1', 'brain2', 'brain3', 'brain4']
    # volumecount = np.ndarray((43, 7))
    # for i, patient_name in enumerate(patient_name_list2):
    #     filedir2 = inputdir2 + '/' + patient_name
    #     filelist2 = os.listdir(filedir2)
    #     filelist2 = sorted(filelist2)
    #     print(filelist2)
    #     file = filelist2[0]
    #     ##取label
    #     print(file)
    #     sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #
    #     # reinforce
    #     volumecount[i][0] = cul_volume(sitkNp, 1, True)
    #     # nero
    #     volumecount[i][1] = cul_volume(sitkNp, 2, True)
    #     # edema
    #     volumecount[i][2] = cul_volume(sitkNp, 3, True) + cul_volume(sitkNp, 4, True) + cul_volume(sitkNp, 5, True)
    #
    #     file = filelist2[1]
    #     print(file)
    #     sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #     volumecount[i][3] = np.count_nonzero(sitkNp)
    #
    #     file = filelist2[2]
    #     print(file)
    #     sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #     volumecount[i][4] = np.count_nonzero(sitkNp)
    #
    #     file = filelist2[3]
    #     print(file)
    #     sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #     volumecount[i][5] = np.count_nonzero(sitkNp)
    #
    #     file = filelist2[4]
    #     print(file)
    #     sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)
    #     sitkNp = sitk.GetArrayFromImage(sitkImage)
    #     volumecount[i][6] = np.count_nonzero(sitkNp)
    #
    # np.savetxt('feature/volume_feature/volumecount_gbm.csv', volumecount)
