import matplotlib.pyplot as plt
import numpy as np

# count data sex and age distribution


gbm = np.load('gbm_sex_age.npy')
gbm = gbm[:, 1:]
gbm = gbm.astype(np.int)

meta = np.load('meta_sex_age.npy')
meta = meta[:, 1:]
meta = meta.astype(np.int)

# gbm_male = np.sum(gbm[:,0])
# meta_male = np.sum(meta[:,0])

# gbm_agemax = np.max(gbm[:, 1])
# gbm_agemin = np.min(gbm[:, 1])
#
# meta_agemax = np.max(meta[:, 1])
# meta_agemin = np.min(meta[:, 1])

gbm_male_count = [0] * 6
gbm_female_count = [0] * 6
for i in range(len(gbm)):
    if gbm[i][0] == 1:
        # male
        gbm_male_count[gbm[i][1] // 10 - 2] += 1
    else:
        # female
        gbm_female_count[gbm[i][1] // 10 - 2] += 1

meta_male_count = [0] * 6
meta_female_count = [0] * 6
for i in range(len(meta)):
    if meta[i][0] == 1:
        # male
        meta_male_count[meta[i][1] // 10 - 2] += 1
    else:
        # female
        meta_female_count[meta[i][1] // 10 - 2] += 1

if __name__ == '__main__':
    N = 6

    # gbm
    men_means = gbm_male_count
    women_means = gbm_female_count

    # meta
    # men_means = meta_male_count
    # women_means = meta_female_count

    labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, men_means, width, label='Male')
    rects2 = ax.bar(x + width / 2, women_means, width, label='Female')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Cases')
    ax.set_xlabel('Age')

    # ax.set_title('GBM')
    ax.set_title('Metastasis')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()
