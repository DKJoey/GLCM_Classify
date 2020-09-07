import os
import shutil
import time

start = time.time()

# copy label or data from preprocess7 to preprocess9


# inputdir1 = 'E:\data\ShandongHospitalBrain_preprocess_7\Brain_metastases_Solitary_Signa_3.0T'
inputdir2 = '/home/cjy/data/ShandongHospitalBrain_preprocess_7/GBM_Signa_3.0T'

# outputdir1 = 'E:\data\ShandongHospitalBrain_preprocess_9\Brain_metastases_Solitary_Signa_3.0T'
outputdir2 = '/home/cjy/data/ShandongHospitalBrain_preprocess_9/GBM_Signa_3.0T'

# patient_name_list1 = os.listdir(inputdir1)
patient_name_list2 = os.listdir(inputdir2)

for patient_name in patient_name_list2:
    filedir2 = inputdir2 + '/' + patient_name
    filelist2 = sorted(os.listdir(filedir2))

    file = filelist2[4]  ##取 DWI 数据
    print(file)
    # DWI: 0   Label: 1   Mask: 2    T1+c: 3    T2: 4   T2-FLAIR: 5
    # ROI
    ## DWI: 0  T1+c: 1   T2-FLAIR: 2   T2: 3
    shutil.copy(filedir2 + '/' + file, outputdir2 + '/' + patient_name + '/' + file)
