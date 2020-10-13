import time

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from utils.load_feature import new_load_feature

X, y, _ = new_load_feature()

# 特征归一化
# X, ranges, minval = autoNorm(X)
X = preprocessing.scale(X)

# pca降维
pca = PCA(n_components=20)
reduced_X = pca.fit_transform(X)

# mrmr降维
# reduced_X = my_mRMR(X, y, 20)
# reduced_X = X

start = time.time()

# grid search
prange = [10 ** c for c in range(-5, 5)]

# tuned_parameters = [{'kernel': ['rbf'], 'gamma': prange, 'C': prange},
#                     {'kernel': ['linear'], 'C': prange}]
tuned_parameters = [{'kernel': ['rbf'], 'gamma': prange, 'C': prange}]

scores = ['precision', 'recall', 'f1', 'accuracy']
# scores = ['f1']

for score in scores:
    clf = GridSearchCV(SVC(), tuned_parameters, scoring=score, cv=10)

    result = cross_val_score(clf, reduced_X, y, scoring=score, cv=5)

    print('%s : %.2f +- %.2f' % (score, np.mean(result) * 100, np.std(result) * 100))

end = time.time()
runtime = end - start
