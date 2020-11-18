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


classifier = ['SVM-RBF', 'kNN', '线性SVM', 'LR', 'SVM-poly2', 'SVM-poly3', 'SVM-poly4', ]
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


plt.rc('font', family='Simhei')

prob2, fpr, tpr, auc = result_auc(result2)
plt.plot(fpr, tpr, 'k:', label='k最近邻:  AUC={0:0.4f}'.format(auc))
#
prob4, fpr, tpr, auc = result_auc(result4)
plt.plot(fpr, tpr, 'k--', label='逻辑回归: AUC={0:0.4f}'.format(auc))

prob3, fpr, tpr, auc = result_auc(result3)
plt.plot(fpr, tpr, 'k-.', label='线性SVM:  AUC={0:0.4f}'.format(auc))
#
prob5, fpr, tpr, auc = result_auc(result5)
# plt.plot(fpr, tpr, label='SVM-Poly2:  AUC={0:0.4f}'.format(auc))
#
prob6, fpr, tpr, auc = result_auc(result6)
# plt.plot(fpr, tpr, label='SVM-Poly3:  AUC={0:0.4f}'.format(auc))
#
prob7, fpr, tpr, auc = result_auc(result7)
# plt.plot(fpr, tpr, label='SVM-Poly4:  AUC={0:0.4f}'.format(auc))

prob1, fpr, tpr, auc = result_auc(result1)
plt.plot(fpr, tpr, 'k-', label='RBF核SVM: AUC={0:0.4f}'.format(auc))

plt.xlabel('1-特异性', fontsize=18)
plt.ylabel('敏感性', fontsize=18)
plt.ylim([0.0, 1.05])
plt.xlim([-0.02, 1.0])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.grid(True)
plt.title('受试者工作特征曲线', fontsize=18)
plt.legend(loc="best", fontsize=18)
plt.savefig('results/auc.jpg')
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

    # plt.bar(range(len(num_list)), num_list, label='boy', fc='y')
    # plt.bar(range(len(num_list)), num_list1, bottom=num_list, label='girl', tick_label=name_list, fc='r')

    x = list(range(len(x_data)))
    total_width, n = 0.8, 2
    width = total_width / n
    #
    plt.bar(x, y_data3, width=width, label='转移瘤 (男)', color='brown')
    plt.bar(x, y_data5, width=width, label='转移瘤 (女)', color='coral', bottom=y_data3)

    for a, b in zip(x, y_data):
        plt.text(a, b + 0.1, '%.d' % b, ha='center', va='bottom', fontsize=18)

    for i in range(len(x)):
        x[i] = x[i] + width

    plt.bar(x, y_data4, width=width, label='胶质瘤 (男)', color='navy', tick_label=x_data)
    plt.bar(x, y_data6, width=width, label='胶质瘤 (女)', color='cyan', bottom=y_data4, tick_label=x_data)

    for a, b in zip(x, y_data2):
        plt.text(a, b + 0.1, '%.d' % b, ha='center', va='bottom', fontsize=18)

    # plt.bar(x_data, y_data, color='red', label='Metastatsis')
    # plt.plot(x_data, y_data3, color='brown', label='Metastatsis (Male)')
    # plt.plot(x_data, y_data5, color='coral', label='Metastatsis (Female)')

    # plt.bar(x_data, y_data2, color='blue', label='Glioma', bottom=y_data)
    # plt.plot(x_data, y_data4, color='navy', label='Glioma (Male)')
    # plt.plot(x_data, y_data6, color='cyan', label='Glioma (Female)')
    plt.title(classifier[p], fontsize=18)
    plt.legend(loc='best', fontsize=12)
    plt.xlabel('概率(%)', fontsize=18)
    plt.xticks(fontsize=13)
    plt.ylim([0, 20])
    plt.ylabel('病例', fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('results/' + str(p) + '.jpg')
    plt.show()
