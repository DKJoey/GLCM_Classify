import numpy as np

classifier = ['rbf', 'knn', 'LinearSVC', 'LR', 'poly']
c_index = classifier[3]
# c_index : 0 1 2 3 4
result = {}

# result_dir = 'results'
result_dir = 'whole_results'

dic = np.load(result_dir + '/dict_' + c_index + '.npy', allow_pickle=True)

dic = dic.item()
for key in dic.keys():
    # print(np.mean(dic[key]))
    result[key] = np.mean(dic[key])

sexage1 = np.load('meta_sex_age.npy')
sexage2 = np.load('gbm_sex_age.npy')
sexage = np.vstack((sexage1, sexage2))

sexage = np.hstack((sexage, np.zeros((88, 1))))

for key in result.keys():
    for i in range(88):
        if sexage[i, 0] == key:
            sexage[i, 3] = result[key]

np.savetxt(result_dir + '/csv/result_' + c_index + '.csv', sexage, fmt='%s')
np.save(result_dir + '/npy/result_' + c_index + '.npy', sexage)

# # male_accuracy
# male_label=[]
# male_prob=[]
# for i in range(88):
#     if int(sexage[i,1])==0:
#         #male:1 female:0
#         if i<45:
#             male_label.append(0)
#         else:
#             male_label.append(1)
#         if float (sexage[i,3]) <0.5:
#             male_prob.append(0)
#         else:
#             male_prob.append(1)
# male_accuracy= accuracy_score(male_label,male_prob)

# age_accuracy
# age_label = []
# age_prob = []
# for i in range(88):
#     if int(sexage[i, 2]) // 10 == 7:
#         # 2 3 4 5 6 7
#         if i < 45:
#             age_label.append(0)
#         else:
#             age_label.append(1)
#         if float(sexage[i, 3]) < 0.5:
#             age_prob.append(0)
#         else:
#             age_prob.append(1)
# age_accuracy = accuracy_score(age_label, age_prob)
