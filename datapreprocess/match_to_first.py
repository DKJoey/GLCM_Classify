import os

import SimpleITK as sitk


def data_resample(_volume: sitk.Image, _new_spacing: tuple):
    _resampleFilter = sitk.ResampleImageFilter()
    _original_size = _volume.GetSize()
    _original_spacing = _volume.GetSpacing()
    _new_size = [int(round(_original_size[0] * (_original_spacing[0] / _new_spacing[0]))),
                 int(round(_original_size[1] * (_original_spacing[1] / _new_spacing[1]))),
                 int(round(_original_size[2] * (_original_spacing[2] / _new_spacing[2])))]

    _resampled = _resampleFilter.Execute(_volume,
                                         _new_size,
                                         sitk.Transform(),
                                         sitk.tkLinear,
                                         _volume.GetOrigin(),
                                         _new_spacing,
                                         _volume.GetDirection(),
                                         0,
                                         _volume.GetPixelID())
    # print_volume_base_info(_resampled)
    return _resampled


root_dir = '/home/cjy/data/comp_pre'
tempdir = os.path.join(root_dir, 'greycompress', 'meta/5_GuHongYu')
tempfiles = sorted(os.listdir(tempdir))
print(tempfiles)

inputdir1 = os.path.join(root_dir, 'greycompress', 'meta')
outputdir1 = os.path.join(root_dir, 'gc_match_to_first', 'meta')
patient_name_list1 = sorted(os.listdir(inputdir1))

inputdir2 = os.path.join(root_dir, 'greycompress', 'GBM')
outputdir2 = os.path.join(root_dir, 'gc_match_to_first', 'GBM')
patient_name_list2 = sorted(os.listdir(inputdir2))

for patient_name in patient_name_list1:
    filedir1 = inputdir1 + '/' + patient_name
    filelist1 = sorted(os.listdir(filedir1))
    i = 0
    for file in filelist1[1:5]:
        i += 1
        sitkImage = sitk.ReadImage(inputdir1 + '/' + patient_name + '/' + file)
        #
        #         # sitkNp = sitk.GetArrayFromImage(sitkImage)
        #         # sitkNp = grayCompression(sitkNp)
        #         # sitkNp_int = sitkNp.astype(np.uint8)
        #         # sitkImage_int = sitk.GetImageFromArray(sitkNp_int)
        #         # sitkImage_int.CopyInformation(sitkImage)
        #
        #
        tempimage = sitk.ReadImage(os.path.join(tempdir, tempfiles[i]))
        filter = sitk.HistogramMatchingImageFilter()
        print(tempfiles[i], file)
        sitkImage = filter.Execute(sitkImage, tempimage)
        if not os.path.exists(outputdir1 + '/' + patient_name):
            os.mkdir(outputdir1 + '/' + patient_name)
        sitk.WriteImage(sitkImage, outputdir1 + '/' + patient_name + '/' + file)

for patient_name in patient_name_list2:
    filedir2 = inputdir2 + '/' + patient_name
    filelist2 = sorted(os.listdir(filedir2))
    i = 0
    for file in filelist2[1:5]:
        i += 1
        sitkImage = sitk.ReadImage(inputdir2 + '/' + patient_name + '/' + file)

        # sitkNp = sitk.GetArrayFromImage(sitkImage)
        # sitkNp = grayCompression(sitkNp)
        # sitkNp_int = sitkNp.astype(np.uint8)
        # sitkImage_int = sitk.GetImageFromArray(sitkNp_int)
        # sitkImage_int.CopyInformation(sitkImage)

        tempimage = sitk.ReadImage(os.path.join(tempdir, tempfiles[i]))
        filter = sitk.HistogramMatchingImageFilter()
        print(tempfiles[i], file)
        sitkImage = filter.Execute(sitkImage, tempimage)
        if not os.path.exists(outputdir2 + '/' + patient_name):
            os.mkdir(outputdir2 + '/' + patient_name)
        sitk.WriteImage(sitkImage, outputdir2 + '/' + patient_name + '/' + file)
