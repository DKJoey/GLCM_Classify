import skimage.feature
import numpy as np
import SimpleITK as sitk
from preprocess import crop, grayCompression

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


if __name__ == '__main__':
    sitkImage = sitk.ReadImage('E:\data\ShandongHospitalBrain_preprocess_7\Metastases_Tumor/5_GuHongYu\DWI.nii.gz')
    sitkNp = sitk.GetArrayFromImage(sitkImage)
    # 裁剪到一样大小
    sitkNp = crop(sitkNp, 123, 220, 220)
    sitkNp = grayCompression(sitkNp)

    sitkNp_int = sitkNp.astype(np.uint8)

    for i in range(sitkNp_int.shape[0]):
        i = 100
        image = sitkNp_int[i, :, :]
        # print(image)
        if (image == np.zeros(image.shape)).all():
            continue
        print(i)
        temp, g = glcm(image)
        break

    g1 = g[:, :, 0, 0]
    g2 = g[:, :, 0, 1]
    g3 = g[:, :, 0, 2]
    g4 = g[:, :, 0, 3]

    print(g[:, :, 0, 0])
    print(g[:, :, 0, 1])
    print(g[:, :, 0, 2])
    print(g[:, :, 0, 3])
    print(temp)

    np1 = sitkNp_int[76, :, :]
    np2 = sitkNp[76, :, :]

    print(sitkNp_int[76, :, :])
    print(sitkNp[76, :, :])