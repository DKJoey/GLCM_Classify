import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

X1 = np.load('feature/v5/X_DWI_transverse.npy')
X2 = np.load('feature/v5/X_DWI_sagittal.npy')
X3 = np.load('feature/v5/X_DWI_coronal.npy')
X4 = np.load('feature/v5/X_T1+c_transverse.npy')
X5 = np.load('feature/v5/X_T1+c_sagittal.npy')
X6 = np.load('feature/v5/X_T1+c_coronal.npy')
X7 = np.load('feature/v5/X_T2_transverse.npy')
X8 = np.load('feature/v5/X_T2_sagittal.npy')
X9 = np.load('feature/v5/X_T2_coronal.npy')
X10 = np.load('feature/v5/X_T2-FLAIR_transverse.npy')
X11 = np.load('feature/v5/X_T2-FLAIR_sagittal.npy')
X12 = np.load('feature/v5/X_T2-FLAIR_coronal.npy')

X = np.hstack((X1, X4, X7, X10))
XA = X1
XB = X4

y = np.load('feature/v5/y.npy')
y = y.ravel()
# y[y==0] = -1

X = preprocessing.scale(X)
XA = preprocessing.scale(XA)
start = time.time()

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
# grid search
gamma_range = [5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1, 5]
cv_scores = []
for g in gamma_range:
    classifier = SVC(kernel="rbf", probability=True, gamma=g, tol=1e-3)
    classifier.fit(X_train, y_train)
    scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
    cv_scores.append(scores.mean())

index = cv_scores.index(max(cv_scores))
g = gamma_range[index]
classifier = SVC(kernel="rbf", probability=True, gamma=g)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# 计算f1
print('f1:')
print(f1_score(y_test, y_pred))
print('acc:')
print(accuracy_score(y_test, y_pred))

# -------------------------------------
XA_train, XA_test, y_train, y_test = train_test_split(XA, y, test_size=1 / 3, random_state=0)
# grid search
gamma_range = [5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1, 5]
cv_scores = []
for g in gamma_range:
    classifierA = SVC(kernel="rbf", probability=True, gamma=g, tol=1e-3)
    classifierA.fit(XA_train, y_train)
    scores = cross_val_score(classifierA, XA_train, y_train, cv=5, scoring='f1')
    cv_scores.append(scores.mean())

index = cv_scores.index(max(cv_scores))
g = gamma_range[index]
classifierA = SVC(kernel="rbf", probability=True, gamma=g)
classifierA.fit(XA_train, y_train)

y_pred = classifierA.predict(XA_test)

# 计算f1
print('f1:')
print(f1_score(y_test, y_pred))
print('acc:')
print(accuracy_score(y_test, y_pred))
# -------------------------------------
XB_train, XB_test, y_train, y_test = train_test_split(XB, y, test_size=1 / 3, random_state=0)
# grid search
gamma_range = [5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1, 5]
cv_scores = []
for g in gamma_range:
    classifierB = SVC(kernel="rbf", probability=True, gamma=g, tol=1e-3)
    classifierB.fit(XB_train, y_train)
    scores = cross_val_score(classifierB, XB_train, y_train, cv=5, scoring='f1')
    cv_scores.append(scores.mean())

index = cv_scores.index(max(cv_scores))
g = gamma_range[index]
classifierB = SVC(kernel="rbf", probability=True, gamma=g)
classifierB.fit(XB_train, y_train)

y_pred = classifierB.predict(XB_test)

# 计算f1
print('f1:')
print(f1_score(y_test, y_pred))
print('acc:')
print(accuracy_score(y_test, y_pred))

display = plot_roc_curve(classifier, X_test, y_test, name='transverse')
ax = plt.gca()
display1 = plot_roc_curve(classifierA, XA_test, y_test, ax=ax, alpha=0.8, name='DWI_transverse')
display2 = plot_roc_curve(classifierB, XB_test, y_test, ax=ax, alpha=0.8, name='T1+c_transverse')
# display3 = plot_roc_curve(classifier, X_train, y_train,ax= ax, alpha= 0.8, name= 'train')
# display.plot(ax=ax, alpha=0.8)
plt.show()

end = time.time()
print('runtime:')
print(end - start)
