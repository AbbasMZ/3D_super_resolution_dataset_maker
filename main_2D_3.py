import numpy as np
import nibabel as nib
import argparse
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('-if', '--input_format', default='.nii.gz', type=str, choices=['.nii.gz', '.nii'], dest='input_format')
parser.add_argument('-ip', '--input_path', default='/run/media/abbas/1TB_E/Study/Vision/3d_super_resolution/generating_LR_from_HR/T1/', type=str, dest='input_path')
parser.add_argument('-of', '--output_format', default='.nii.gz', type=str, choices=['.nii.gz', '.nii'], dest='output_format')
parser.add_argument('-opl', '--output_path_LR', default='/run/media/abbas/1TB_E/Study/Vision/3d_super_resolution/generating_LR_from_HR/T1_2D_LR/', type=str, dest='output_path_LR')
parser.add_argument('-oph', '--output_path_HR', default='/run/media/abbas/1TB_E/Study/Vision/3d_super_resolution/generating_LR_from_HR/T1_2D_HR/', type=str, dest='output_path_HR')
""
args = parser.parse_args()

##### Retrieve list of files in the input path
files = os.listdir(args.input_path)
# print(files)
# exit(0)

##### Loop over input files
for f in files:

    file_name = f[:-len(args.input_format)]
    print('starting file: ', f)
    ##### load 3D image as HR
    data = nib.load(args.input_path + file_name + args.input_format)
    # print("data: ", data.shape)


    ##### float64
    fdata = data.get_fdata()
    # print("fdata: ", fdata.shape)
    # print("fdata: ", fdata.dtype)


    ##### Loop over axial dimension of each input file
    for index in range(fdata.shape[1]):
        print('starting index ', index)
        ##### 2D version
        fdata_2D = fdata[:,index,:]
        # print("fdata_2D: ", fdata_2D.shape)
        # print("fdata_2D: ", fdata_2D.dtype)

        # save image as HR
        save_path = args.output_path_HR + file_name + '_' + str(index) + args.output_format
        nib.save(nib.Nifti1Image(fdata_2D, None, header=data.header.copy()), save_path)


        ##### numpy array
        fdata_np = np.array(fdata_2D)
        # print("fdata_np: ", fdata_np.shape)
        # print("fdata_np: ", fdata_np.dtype)
        HR_img = fdata_np


        ##### fft
        HR_img_fft = np.fft.fftn(HR_img)
        # print("HR_img_fft: ", HR_img_fft.shape)
        # print("HR_img_fft: ", HR_img_fft.dtype)

        # HR_img_fft_real = np.real(HR_img_fft)
        # HR_img_fft_real = (HR_img_fft_real * 255 / np.max(HR_img_fft_real)).astype(np.uint8)
        # Image.fromarray(HR_img_fft_real).save('T1_HR_img_fft_real.png')


        ##### shift
        HR_img_fft_shift = np.fft.fftshift(HR_img_fft)
        # print("HR_img_fft_shift: ", HR_img_fft_shift.shape)
        # print("HR_img_fft_shift: ", HR_img_fft_shift.dtype)

        # HR_img_fft_shift_real = np.real(HR_img_fft_shift)
        # HR_img_fft_shift_real = (HR_img_fft_shift_real * 255 / np.max(HR_img_fft_shift_real)).astype(np.uint8)
        # Image.fromarray(HR_img_fft_shift_real).save('T1_HR_img_fft_shift_real.png')


        ##### crop outside and pad with zeros
        zeros_img = np.zeros((fdata_2D.shape[0], fdata_2D.shape[1]), dtype=np.complex128)
        dim0_start = int(np.floor(fdata_2D.shape[0] / 4))
        dim0_end = int(np.floor(fdata_2D.shape[0] * 3 / 4))
        dim1_start = int(np.floor(fdata_2D.shape[1] / 4))
        dim1_end = int(np.floor(fdata_2D.shape[1] * 3 / 4))
        # print("dim0_start, dim0_end, dim1_start, dim1_end: ", dim0_start, dim0_end, dim1_start, dim1_end)
        zeros_img[dim0_start:dim0_end,dim1_start:dim1_end] = HR_img_fft_shift[dim0_start:dim0_end,dim1_start:dim1_end]
        LR_img_fft_shift_zero_padded = zeros_img
        # print("LR_img_fft_shift_zero_padded: ", LR_img_fft_shift_zero_padded.shape)
        # print("LR_img_fft_shift_zero_padded: ", LR_img_fft_shift_zero_padded.dtype)

        # LR_img_fft_shift_zero_padded_real = np.real(LR_img_fft_shift_zero_padded)
        # LR_img_fft_shift_zero_padded_real = (LR_img_fft_shift_zero_padded_real * 255 / np.max(LR_img_fft_shift_zero_padded_real)).astype(np.uint8)
        # Image.fromarray(LR_img_fft_shift_zero_padded_real).save('T1_LR_img_fft_shift_zero_padded_real.png') 


        ##### inverse shift
        LR_img_fft = np.fft.ifftshift(LR_img_fft_shift_zero_padded)
        # print("LR_img_fft: ", LR_img_fft.shape)
        # print("LR_img_fft: ", LR_img_fft.dtype)

        # LR_img_fft_real = np.real(LR_img_fft)
        # LR_img_fft_real = (LR_img_fft_real * 255 / np.max(LR_img_fft_real)).astype(np.uint8)
        # Image.fromarray(LR_img_fft_real).save('T1_LR_img_fft_real.png')


        ##### inverse fft
        LR_img = np.fft.ifftn(LR_img_fft)
        # print("LR_img: ", LR_img.shape)
        # print("LR_img: ", LR_img.dtype)

        LR_img_real = np.real(LR_img)
        # print("LR_img_real: ", LR_img_real.shape)
        # print("LR_img_real: ", LR_img_real.dtype)


        # save image as LR
        save_path = args.output_path_LR + file_name + '_' + str(index) + args.output_format
        nib.save(nib.Nifti1Image(LR_img_real, None, header=data.header.copy()), save_path)    