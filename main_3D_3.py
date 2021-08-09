import numpy as np
import nibabel as nib
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('-if', '--input_format', default='.nii.gz', type=str, choices=['.nii.gz', '.nii'], dest='input_format')
parser.add_argument('-ip', '--input_path', default='/run/media/abbas/1TB_E/Study/Vision/3d_super_resolution/generating_LR_from_HR/T1/', type=str, dest='input_path')
parser.add_argument('-f', '--factor', default=2, type=int, dest='factor')
parser.add_argument('-d', '--dimension', default='2', choices=['2', '3'], type=str, dest='dimension')
parser.add_argument('-x', '--axis_2D', default=1, choices=[0, 1, 2], type=int, dest='axis_2D')
parser.add_argument('-of', '--output_format', default='.nii.gz', type=str, choices=['.nii.gz', '.nii'], dest='output_format')
parser.add_argument('-opl', '--output_path_LR', default='/run/media/abbas/1TB_E/Study/Vision/3d_super_resolution/generating_LR_from_HR/T1_2D_LR/', type=str, dest='output_path_LR')
parser.add_argument('-oph', '--output_path_HR', default='/run/media/abbas/1TB_E/Study/Vision/3d_super_resolution/generating_LR_from_HR/T1_2D_HR/', type=str, dest='output_path_HR')
args = parser.parse_args()

##### Retrieve list of files in the input path
files = os.listdir(args.input_path)

##### Loop over input files
for f in files:

    file_name = f[:-len(args.input_format)]
    print('starting file: ', f)

    
    ##### load image as HR
    data = nib.load(args.input_path + file_name + args.input_format)
    # print("data: ", data.shape)


    ##### float64
    fdata = data.get_fdata()
    # print("fdata: ", fdata.shape)
    # print("fdata: ", fdata.dtype)


    ##### save 2D HR images
    if args.dimension == '2':
        for index in range(fdata.shape[args.axis_2D]):
            ##### 2D version
            if args.axis_2D == 0:
                fdata_2D = fdata[:,index,:]
            elif args.axis_2D == 1:
                fdata_2D = fdata[:,index,:]
            else:
                fdata_2D = fdata[:,:,index]

            # print("fdata_2D: ", fdata_2D.shape)
            # print("fdata_2D: ", fdata_2D.dtype)


            # save image as HR
            Path(args.output_path_HR).mkdir(parents=True, exist_ok=True)
            save_path = args.output_path_HR + file_name + '_' + str(index) + args.output_format
            nib.save(nib.Nifti1Image(fdata_2D, None, header=data.header.copy()), save_path)


    ##### numpy array
    fdata_np = np.array(fdata)
    # print("fdata_np: ", fdata_np.shape)
    # print("fdata_np: ", fdata_np.dtype)

    HR_img = fdata_np


    ##### fft
    HR_img_fft = np.fft.fftn(HR_img)
    # print("HR_img_fft: ", HR_img_fft.shape)
    # print("HR_img_fft: ", HR_img_fft.dtype)


    ##### shift
    HR_img_fft_shift = np.fft.fftshift(HR_img_fft)
    # print("HR_img_fft_shift: ", HR_img_fft_shift.shape)
    # print("HR_img_fft_shift: ", HR_img_fft_shift.dtype)

    ##### crop outside and pad with zeros
    zeros_img = np.zeros((fdata.shape[0], fdata.shape[1], fdata.shape[2]), dtype=np.complex128)
    dim0_start = int(np.floor(fdata.shape[0] * (args.factor - 1) / (2 * args.factor)))
    dim0_end = int(np.floor(fdata.shape[0] * (args.factor + 1) / (2 * args.factor)))
    dim1_start = int(np.floor(fdata.shape[1] * (args.factor - 1) / (2 * args.factor)))
    dim1_end = int(np.floor(fdata.shape[1] * (args.factor + 1) / (2 * args.factor)))
    dim2_start = int(np.floor(fdata.shape[2] * (args.factor - 1) / (2 * args.factor)))
    dim2_end = int(np.floor(fdata.shape[2] * (args.factor + 1) / (2 * args.factor)))
    # print("dim0_start, dim0_end, dim1_start, dim1_end, dim2_start, dim2_end: ", dim0_start, dim0_end, dim1_start, dim1_end, dim2_start, dim2_end)
    zeros_img[dim0_start:dim0_end, dim1_start:dim1_end, dim2_start:dim2_end] = HR_img_fft_shift[dim0_start:dim0_end, dim1_start:dim1_end, dim2_start:dim2_end]
    LR_img_fft_shift_zero_padded = zeros_img
    # print("LR_img_fft_shift_zero_padded: ", LR_img_fft_shift_zero_padded.shape)
    # print("LR_img_fft_shift_zero_padded: ", LR_img_fft_shift_zero_padded.dtype)

    ##### inverse shift
    LR_img_fft = np.fft.ifftshift(LR_img_fft_shift_zero_padded)
    # print("LR_img_fft: ", LR_img_fft.shape)
    # print("LR_img_fft: ", LR_img_fft.dtype)

    ##### inverse fft
    LR_img = np.fft.ifftn(LR_img_fft)
    # print("LR_img: ", LR_img.shape)
    # print("LR_img: ", LR_img.dtype)

    LR_img_real = np.real(LR_img)
    # print("LR_img_real: ", LR_img_real.shape)
    # print("LR_img_real: ", LR_img_real.dtype)



    # save image as LR
    Path(args.output_path_LR).mkdir(parents=True, exist_ok=True)
    if args.dimension == '3':
        save_path = args.output_path_LR + file_name + '_' + str(index) + args.output_format
        nib.save(nib.Nifti1Image(LR_img_real, None, header=data.header.copy()), save_path)    
    else:
        for index in range(LR_img_real.shape[args.axis_2D]):
            ##### 2D version
            if args.axis_2D == 0:
                LR_img_real_2D = LR_img_real[:,index,:]
            elif args.axis_2D == 1:
                LR_img_real_2D = LR_img_real[:,index,:]
            else:
                LR_img_real_2D = LR_img_real[:,:,index]
            # print("LR_img_real_2D: ", LR_img_real_2D.shape)
            # print("LR_img_real_2D: ", LR_img_real_2D.dtype)

            # save image as HR
            save_path = args.output_path_LR + file_name + '_' + str(index) + args.output_format
            nib.save(nib.Nifti1Image(LR_img_real_2D, None, header=data.header.copy()), save_path)