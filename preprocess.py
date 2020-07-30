import numpy as np
import skimage.morphology as sm

# useful function in preprocess


# 将三维image从中心裁剪
# image size A*B*C and A>=a B>=b C>=c
def crop(image, a, b, c):
    A = image.shape[0]
    B = image.shape[1]
    C = image.shape[2]
    return image[A // 2 - a // 2:A // 2 - a // 2 + a, B // 2 - b // 2:B // 2 - b // 2 + b,
           C // 2 - c // 2:C // 2 - c // 2 + c]


def autoNorm(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    range = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(range, (m, 1))
    return normData, range, minVals


def grayCompression(mat):
    maxmat = np.max(mat)
    minmat = np.min(mat)
    mat = 255 * (mat-minmat) / (maxmat-minmat)
    return mat


def contour(image):
    image[image != 0] = 1
    big = sm.dilation(image, sm.cube(5))
    small = sm.erosion(image, sm.cube(5))
    res = big - small
    return res


if __name__ == '__main__':
    image = np.arange(27)
    # print(image)
    # image = image.reshape(3, 3, 3)
    # print(image)
    # print(crop(image, 3, 3, 3))
    # mat = np.array([[[-1, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=float)
    # mat = np.random.random((2,2,2))
    # mat1 = grayCompression(mat)
    # print(mat1)
    # mat2 = mat1.astype(np.uint8)


