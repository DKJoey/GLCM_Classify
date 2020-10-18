import time

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils.feature import feature_select
from utils.load_feature import new_load_feature

X, y, namesex = new_load_feature()

# X, y = new_load_feature()

results = np.zeros((1000, 4))

name_results = {}

# 特征归一化
# X, ranges, minval = autoNorm(X)
X = preprocessing.scale(X)

reduced_X = feature_select(X, y, 20, 'rank')

for i in range(1000):
    print(i)

    start = time.time()

    # 数据集划分
    X_train, X_test, y_train, y_test, _, namesex_test = train_test_split(reduced_X, y, namesex, test_size=1 / 5,
                                                                         stratify=y)

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

for key in name_results.keys():
    name_results[key] = np.mean(name_results[key])

sexage1 = np.load('../meta_sex_age.npy')
sexage2 = np.load('../gbm_sex_age.npy')
sexage = np.vstack((sexage1, sexage2))
sexage = np.hstack((sexage, np.zeros((88, 1))))

for key in name_results.keys():
    for i in range(88):
        if sexage[i, 0] == key:
            sexage[i, 3] = name_results[key]

np.savetxt('../results/linear.csv', sexage, fmt='%s')
np.save('../results/linear.npy', sexage)
