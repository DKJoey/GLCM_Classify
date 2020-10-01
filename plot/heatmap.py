import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse

# feature heatmap
from utils.load_feature import new_load_feature


def corrplot(data, pvalues, labels):
    """Creates a correlation plot of the passed data.
    The function returns the plot which can then be shown with
    plot.show(), saved to a file with plot.savefig(), or manipulated
    in any other standard matplotlib way.
    data is the correlation matrix, a 2-D numpy array containing
    the pairwise correlations between variables;
    pvalues is a matrix containing the pvalue for each corresponding
    correlation value; if none it is assumed to be the zero matrix
    labels is an array containing the variable names
    """

    plt.figure(1)

    column_labels = labels
    row_labels = labels

    ax = plt.subplot(1, 1, 1, aspect='equal')

    height, width = data.shape
    num_cols, num_rows = width, height

    if pvalues is None:
        pvalues = np.zeros([num_rows, num_cols])

    shrink = 0.9

    print(data.shape)
    print(pvalues.shape)

    poscm = cm.get_cmap('Blues')
    negcm = cm.get_cmap('Oranges')

    for x in range(height):
        for y in range(width):
            d = data[x, y]
            c = pvalues[x, y]
            rotate = -45 if d > 0 else +45
            clrmap = poscm if d >= 0 else negcm
            d_abs = np.abs(d)
            ellipse = Ellipse((x, y),
                              width=1 * shrink,
                              height=(shrink - d_abs * shrink),
                              angle=rotate)
            ellipse.set_edgecolor('black')
            ellipse.set_facecolor(clrmap(d_abs))
            if c > 0.05:
                ellipse.set_linestyle('dotted')
            ax.add_artist(ellipse)

    ax.set_xlim(-1, num_cols)
    ax.set_ylim(-1, num_rows)

    ax.xaxis.tick_top()
    xtickslocs = np.arange(len(row_labels))
    ax.set_xticks(xtickslocs)
    ax.set_xticklabels(row_labels, rotation=30, fontsize='small', ha='left')

    ax.invert_yaxis()
    ytickslocs = np.arange(len(row_labels))
    ax.set_yticks(ytickslocs)
    ax.set_yticklabels(column_labels, fontsize='small')

    return plt


if __name__ == "__main__":
    sexage1 = np.load('../meta_sex_age.npy')
    sexage2 = np.load('../gbm_sex_age.npy')
    sexage = np.vstack((sexage1, sexage2))
    sexage = sexage[:, 1:]
    sexage = sexage.astype(np.int)

    # different ori
    # X = np.hstack((X1, X2, X3))
    # different modal
    # X = np.hstack((X1, X4, X7, X10))

    # X = preprocessing.scale(X)
    # X1 = preprocessing.scale(X1)

    # y = np.load('../feature/v5/y.npy')
    # y = y.reshape((88, 1))

    # data = np.hstack((y, X))
    # data = np.hstack((y, sexage))
    X, y = new_load_feature()

    data = X
    print(data.shape)

    # label = ['label', 'sex', 'age']
    pdata = pd.DataFrame(data)
    # print(pdata.columns)

    corrmat = pd.DataFrame(data).corr()
    corrmat = abs(corrmat)
    # np.savetxt('corrmat.csv', corrmat.values)

    f, ax = plt.subplots()
    sns.heatmap(corrmat, square=True, cmap="rainbow", vmax=1, vmin=0, xticklabels=False,
                yticklabels=False)
    plt.show()

    # k = 5
    # cols = corrmat.nlargest(k, 0)[0].index
    # cm = np.corrcoef(pdata[cols].values.T)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(cm, annot=True, yticklabels=cols.values,xticklabels=cols.values)
    # plt.show()
