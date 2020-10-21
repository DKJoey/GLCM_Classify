import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import skimage.feature

from utils.preprocess import crop, grayCompression


def show_nonzero(g):
    ans = g
    for i in range(ans.shape[0]):
        if np.sum(ans[i, :]) > 0:
            ans = ans[i:, :]
            break

    for i in range(ans.shape[0] - 1, -1, -1):
        if np.sum(ans[i, :]) > 0:
            ans = ans[:i + 1, :]
            break

    for j in range(ans.shape[1]):
        if np.sum(ans[:, j]) > 0:
            ans = ans[:, j:]
            break

    for j in range(ans.shape[1] - 1, -1, -1):
        if np.sum(ans[:, j]) > 0:
            ans = ans[:, :j + 1]
            break

    return ans


dir = '/home/cjy/data/10.18/match/GBM/23_GuoNiNi_Sep'
odir = '/home/cjy/data/10.18/match/plot'

f_list = os.listdir(dir)
f_list = sorted(f_list)

label = f_list[0]
file = f_list[4]
# 0：label    1:DWI    2:t1    3:t2    4:flair    5:mask

slice_ori = 2

sitkImage = sitk.ReadImage(os.path.join(dir, file))
labelImage = sitk.ReadImage(os.path.join(dir, label))

sitkNp = sitk.GetArrayFromImage(sitkImage)
labelNp = sitk.GetArrayFromImage(labelImage)

# 裁剪到一样大小
sitkNp = crop(sitkNp, 123, 220, 220)
labelNp = crop(labelNp, 123, 220, 220)

sitkNp = grayCompression(sitkNp)
sitkNp_int = sitkNp.astype(np.uint8)

# mask roi
b_sitkNp_int = sitkNp_int.copy()
sitkNp_int[labelNp == 0] = 0


feature = np.zeros((1, 16))
for i in range(sitkNp_int.shape[slice_ori]):
    if slice_ori == 0:
        image = sitkNp_int[i, :, :]
        b_image = b_sitkNp_int[i, :, :]
    elif slice_ori == 1:
        image = sitkNp_int[:, i, :]
        b_image = b_sitkNp_int[:, i, :]
    elif slice_ori == 2:
        image = sitkNp_int[:, :, i]
        b_image = b_sitkNp_int[:, :, i]
    else:
        print("slice_ori error")
        break
    # print(image)
    if (image == np.zeros(image.shape)).all():
        continue
    # temp = glcm(image)
    print(i)

    if not os.path.exists(os.path.join(odir, str(slice_ori), str(i))):
        os.makedirs(os.path.join(odir, str(slice_ori), str(i)))

    plt.imshow(image, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(odir, str(slice_ori), str(i), 'tumor_color.jpg'))
    plt.show()

    plt.imshow(image, cmap='bone')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(odir, str(slice_ori), str(i), 'tumor.jpg'))
    plt.show()

    plt.imshow(b_image, cmap='bone')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(odir, str(slice_ori), str(i), 'brain.jpg'))
    plt.show()

    g = skimage.feature.greycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    # print(g.shape)

    # plt.imshow(g[1:, 1:, 0, 0], cmap='jet')
    plt.imshow(show_nonzero(g[1:, 1:, 0, 0]), cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(odir, str(slice_ori), str(i), 'theta0.jpg'))
    plt.show()
    plt.imshow(show_nonzero(g[1:, 1:, 0, 1]), cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(odir, str(slice_ori), str(i), 'theta45.jpg'))
    plt.show()
    plt.imshow(show_nonzero(g[1:, 1:, 0, 2]), cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(odir, str(slice_ori), str(i), 'theta90.jpg'))
    plt.show()
    plt.imshow(show_nonzero(g[1:, 1:, 0, 3]), cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(odir, str(slice_ori), str(i), 'theta135.jpg'))
    plt.show()

    contrast = skimage.feature.greycoprops(g, prop='contrast')
    correlation = skimage.feature.greycoprops(g, prop='correlation')
    energy = skimage.feature.greycoprops(g, prop='energy')
    homogeneity = skimage.feature.greycoprops(g, prop='homogeneity')
    temp = np.hstack((contrast, correlation, energy, homogeneity))

    f = plt.figure()  # 生成一个窗体，窗体的规格可以在这里设置
    f.add_subplot(1, 4, 1)  # 规格为2X2  4个子图中的第一个，在左上角---从左到右，从上到下排
    plt.imshow(contrast, cmap='jet')  # 这里显示的是RGB中的R通道的灰度图
    plt.title('contrast')  # 子图的标题
    plt.xticks([]), plt.yticks([])  # 去除坐标轴
    f.add_subplot(1, 4, 2)  # 4个子图中的第2个，右上角
    plt.imshow(correlation, cmap='jet')
    plt.title('correlation')
    plt.xticks([]), plt.yticks([])
    f.add_subplot(1, 4, 3)  # 左下角
    plt.imshow(energy, cmap='jet')
    plt.title('energy')
    plt.xticks([]), plt.yticks([])
    f.add_subplot(1, 4, 4)  # 右下角
    plt.imshow(homogeneity, cmap='jet')
    plt.title('homogeneity')
    plt.xticks([]), plt.yticks([])
    plt.savefig(os.path.join(odir, str(slice_ori), str(i), 'feature.jpg'))
    plt.show()  # 显示

    feature = np.vstack((feature, temp))
# 去除dummyhead
feature = feature[1:, :]
mean_feature = np.mean(feature, axis=0)
contrast = mean_feature[0:4].reshape((1, 4))
correlation = mean_feature[4:8].reshape((1, 4))
energy = mean_feature[8:12].reshape((1, 4))
homogeneity = mean_feature[12:16].reshape((1, 4))

f = plt.figure()  # 生成一个窗体，窗体的规格可以在这里设置
f.add_subplot(1, 4, 1)  # 规格为2X2  4个子图中的第一个，在左上角---从左到右，从上到下排
plt.imshow(contrast, cmap='jet')  # 这里显示的是RGB中的R通道的灰度图
plt.title('contrast')  # 子图的标题
plt.xticks([]), plt.yticks([])  # 去除坐标轴
f.add_subplot(1, 4, 2)  # 4个子图中的第2个，右上角
plt.imshow(correlation, cmap='jet')
plt.title('correlation')
plt.xticks([]), plt.yticks([])
f.add_subplot(1, 4, 3)  # 左下角
plt.imshow(energy, cmap='jet')
plt.title('energy')
plt.xticks([]), plt.yticks([])
f.add_subplot(1, 4, 4)  # 右下角
plt.imshow(homogeneity, cmap='jet')
plt.title('homogeneity')
plt.xticks([]), plt.yticks([])
plt.savefig(os.path.join(odir, str(slice_ori), 'mean_feature.jpg'))
plt.show()  # 显示
