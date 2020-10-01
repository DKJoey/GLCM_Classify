import matplotlib.pyplot  as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def calc_auc(y_pred_proba, labels, exp_run_folder, classifier, fold):
    auc = roc_auc_score(labels, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(labels, y_pred_proba)
    curve_roc = np.array([fpr, tpr])
    # dataile_id = open(exp_run_folder+'/data/roc_{}_{}.txt'.format(classifier, fold), 'w+')
    # np.savetxt(dataile_id, curve_roc)
    # dataile_id.close()
    plt.plot(fpr, tpr, label='ROC curve: AUC={0:0.2f}'.format(auc))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.grid(True)
    plt.title('ROC Fold {}'.format(fold))
    plt.legend(loc="lower left")
    # plt.savefig(exp_run_folder+'/data/roc_{}_{}.pdf'.format(classifier, fold), format='pdf')
    return auc


classifier = ['SVM-RBF', 'kNN', 'SVM-Linear', 'LR', 'SVM-poly2', 'SVM-poly3', 'SVM-poly4', ]
#  result :  name  sex  age  prob
result1 = np.load('results/svm.npy')
result2 = np.load('results/knn.npy')
result3 = np.load('results/linear.npy')
result4 = np.load('results/LR.npy')
result5 = np.load('results/poly2.npy')
result6 = np.load('results/poly3.npy')
result7 = np.load('results/poly4.npy')

result = [result1, result2, result3, result4, result5, result6, result7]


def result_auc(result):
    label1 = np.zeros((45, 1))
    label2 = np.ones((43, 1))
    label = np.vstack((label1, label2))

    prob = np.zeros((88, 1))
    for i in range(88):
        prob[i] = float(result[i, 3])

    fpr, tpr, thresholds = roc_curve(label, prob)
    auc = roc_auc_score(label, prob)

    return prob, fpr, tpr, auc


prob2, fpr, tpr, auc = result_auc(result2)
plt.plot(fpr, tpr, label='kNN:            AUC={0:0.4f}'.format(auc))
#
prob4, fpr, tpr, auc = result_auc(result4)
plt.plot(fpr, tpr, label='LR:               AUC={0:0.4f}'.format(auc))

prob3, fpr, tpr, auc = result_auc(result3)
plt.plot(fpr, tpr, label='SVM-Linear: AUC={0:0.4f}'.format(auc))
#
prob5, fpr, tpr, auc = result_auc(result5)
plt.plot(fpr, tpr, label='SVM-poly2:  AUC={0:0.4f}'.format(auc))
#
prob6, fpr, tpr, auc = result_auc(result6)
plt.plot(fpr, tpr, label='SVM-poly3:  AUC={0:0.4f}'.format(auc))
#
prob7, fpr, tpr, auc = result_auc(result7)
plt.plot(fpr, tpr, label='SVM-poly4:  AUC={0:0.4f}'.format(auc))

prob1, fpr, tpr, auc = result_auc(result1)
plt.plot(fpr, tpr, label='SVM-RBF:    AUC={0:0.4f}'.format(auc))

plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
# plt.grid(True)
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

x_data = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']

prob = [prob1, prob2, prob3, prob4, prob5, prob6, prob7]

for p in range(len(prob)):
    y_data = [0] * 10  # meta
    y_data2 = [0] * 10  # gbm
    y_data3 = [0] * 10  # meta male
    y_data4 = [0] * 10  # gbm male
    y_data5 = [0] * 10  # meta female
    y_data6 = [0] * 10  # gbm female
    for i in range(88):
        if i < 45:
            y_data[int(prob[p][i] * 10) if int(prob[p][i] * 10) < 10 else 9] += 1
            if result[p][i, 1] == '1':
                y_data3[int(prob[p][i] * 10) if int(prob[p][i] * 10) < 10 else 9] += 1
            else:
                y_data5[int(prob[p][i] * 10) if int(prob[p][i] * 10) < 10 else 9] += 1
        else:
            y_data2[int(prob[p][i] * 10) if int(prob[p][i] * 10) < 10 else 9] += 1
            if result[p][i, 1] == '1':
                y_data4[int(prob[p][i] * 10) if int(prob[p][i] * 10) < 10 else 9] += 1
            else:
                y_data6[int(prob[p][i] * 10) if int(prob[p][i] * 10) < 10 else 9] += 1

    plt.plot(x_data, y_data, color='red', label='metastatsis')
    plt.plot(x_data, y_data3, color='brown', label='metastatsis (male)')
    plt.plot(x_data, y_data5, color='coral', label='metastatsis (female)')

    plt.plot(x_data, y_data2, color='blue', label='glioma')
    plt.plot(x_data, y_data4, color='navy', label='glioma (male)')
    plt.plot(x_data, y_data6, color='cyan', label='glioma (female)')
    plt.title(classifier[p])
    plt.legend(loc='best')
    plt.xlabel('Prob')
    plt.ylabel('Cases')
    plt.show()
