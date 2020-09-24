import time

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from load_feature import load_feature

X, y, namesex = load_feature()
results = np.zeros((1000, 4))

name_results = {}

# y = y.reshape((88, 1))
# y = np.hstack((name, sex, y))
# XB = np.hstack((X_B1cc,X_B1cs,X_B1ct))
# yB = np.load('feature/BraTS/y.npy')
# yB[yB == 0] = 1
# X = np.vstack((X,XB))
# y = np.vstack((y,yB))
#
# y[y==0] = -1
#
# 特征归一化
# X, ranges, minval = autoNorm(X)
X = preprocessing.scale(X)

# # pca降维
pca = PCA(n_components=20)
reduced_X = pca.fit_transform(X)

# mrmr降维
# reduced_X = my_mRMR(X, y, 20)

for i in range(1000):
    print(i)

    start = time.time()

    # 数据集划分
    X_train, X_test, y_train, y_test, _, namesex_test = train_test_split(reduced_X, y, namesex, test_size=1 / 5,
                                                                         stratify=y)
    # grid search
    # y_train = y_train[:, 2]
    # y_test = y_test[:, 2]

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)

    # predict the result
    y_pred = classifier.predict(X_test)

    # # 查看每种类别的概率
    y_pred_proba = classifier.predict_proba(X_test)
    # params = classifier.get_params()

    for j in range(18):
        if namesex_test[j, 0] not in name_results.keys():
            name_results[namesex_test[j, 0]] = []
        name_results[namesex_test[j, 0]].append(y_pred_proba[j, 1])

    # 计算f1、accuracy
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    end = time.time()
    runtime = end - start

    results[i, 0] = i
    results[i, 1] = f1
    results[i, 2] = acc
    results[i, 3] = runtime

# print(results[:,1])

f1mean = np.mean(results[:, 1])
f1std = np.std(results[:, 1])
accmean = np.mean(results[:, 2])
accstd = np.std(results[:, 2])
runtimemean = np.mean(results[:, 3])
runtimestd = np.std(results[:, 3])

print('%.3f +- %.3f' % (accmean, accstd))
print('%.3f +- %.3f' % (f1mean, f1std))
print('%.3f +- %.3f' % (runtimemean, runtimestd))

# for key in name_results.keys():
#     name_results[key] = mean(name_results[key])
np.save('whole_results/dict_LinearSVC.npy', name_results)
