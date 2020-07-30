import skimage.feature
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from preprocess import crop, grayCompression


def neighbour_glcm(image):
    g = skimage.feature.greycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    # 对比度
    # contrast = skimage.feature.greycoprops(g, prop='contrast')
    # mean = np.mean(contrast)
    # 非相似性
    # dissimilarity = skimage.feature.greycoprops(g, prop='dissimilarity')
    # 同质度
    # homogeneity = skimage.feature.greycoprops(g, prop='homogeneity')
    # mean = np.mean(homogeneity)
    # 能量
    energy = skimage.feature.greycoprops(g, prop='energy')
    mean = np.mean(energy)
    # 相关性
    # correlation = skimage.feature.greycoprops(g, prop='correlation')
    # mean = np.mean(correlation)
    # 角二阶矩
    # ASM = skimage.feature.greycoprops(g, prop='ASM')
    return mean


def get_neighbour(image, x, y, kernel):
    # padding
    d = (kernel - 1) // 2
    image = np.pad(image, ((d, d), (d, d)))

    return image[x - d:x + d + 1, y - d:y + d + 1]


if __name__ == '__main__':
    # sitkImage = sitk.ReadImage('E:\data\ShandongHospitalBrain_preprocess_9\Metastases_Tumor/48_GuanWenTian/T1+c.nii.gz')
    sitkImage = sitk.ReadImage('E:\data\ShandongHospitalBrain_preprocess_9\GBM_Tumor/66_WangXiuZhen/T1+c.nii.gz')
    sitkNp = sitk.GetArrayFromImage(sitkImage)
    # 裁剪到一样大小
    sitkNp = crop(sitkNp, 123, 220, 220)
    sitkNp = grayCompression(sitkNp)
    sitkNp_int = sitkNp.astype(np.uint8)
    # tumor slice z loc
    image = sitkNp_int[67, :, :]

    plt.imshow(image, cmap='bone')
    plt.show()

    glcmNp = np.zeros(image.shape)
    for i in range(220):
        for j in range(220):
            if image[i, j] != 17:
                neighbour = get_neighbour(image, i, j, 5)
                glcmNp[i, j] = neighbour_glcm(neighbour)

    plt.imshow(glcmNp, cmap='plasma', vmin=0, vmax=1)
    plt.show()
