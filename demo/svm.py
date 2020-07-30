import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # 支持向量机的分类器
# 利用sklearn 随机生成数据
from sklearn.datasets import make_blobs, make_circles


def plot_svc_decision_function(model, ax=None, plot_support=True):
    "绘制二维的决定方程"
    if ax is None:
        ax = plt.gca()  # 绘制一个子图对象
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(xlim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # 绘制决策边界
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # 绘制支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# -----------------------------
# 线性SVM
# x,y = make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.6)
# plt.scatter(x[:,0],x[:,1],c=y,cmap='autumn')
# plt.show()
#
#
# model = SVC(kernel='linear')
# model.fit(x,y)
#
# plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='autumn')
# plot_svc_decision_function(model)
# plt.show()


# ------------------------------------
# RBF 径向基函数
# X, y = make_circles(100, factor=0.2, noise=0.1, shuffle=True)
#
# clf = SVC(kernel='rbf', C=1E6).fit(X, y)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(clf, plot_support=False)
# plt.show()


# --------------------------------
# 调节svm参数：soft margin问题
# c大 严格
# c小 错误容忍大
# X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plt.show()
#
# # 观察C 参数的大小对结果的影响
# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# for axi, C in zip(ax, [20, 0.05]):
#     model = SVC(kernel='linear', C=C).fit(X, y)
#     axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#     plot_svc_decision_function(model, axi)
#     axi.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, lw=1, facecolors='none')
#     axi.set_title('C={0:.1f}'.format(C), size=14)
# fig.show()
# # 距离越小，泛化能力越差


# --------------------------------
# 观察gamma值的影响,gamma越大，模型越复杂，泛化能力越差
# X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.1)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# fig, ax = plt.subplots(1, 3, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# for axi, gamma in zip(ax, [10.0,1, 0.1]):
#     model = SVC(kernel='rbf', gamma=gamma).fit(X, y)
#     axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#     plot_svc_decision_function(model, axi)
#     axi.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, lw=1, facecolors='none')
#     axi.set_title('gamma={0:.1f}'.format(gamma), size=14)
# fig.show()
