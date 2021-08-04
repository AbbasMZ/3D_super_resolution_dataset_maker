from re import A
import numpy as np
import scipy.fft as spfft
import nibabel as nib
import torch.nn
import torch
from PIL import Image

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# img = mpimg.imread('sample.jpeg')



##### load image as HR
path = "/run/media/abbas/1TB_E/Study/Vision/3d_super_resolution/generating_LR_from_HR/"
path_ext = "T1/"
load_path = path + path_ext + "IXI012-HH-1211-T1" + ".nii.gz"

data = nib.load(load_path)
print("data: ", data.shape)
# print("data: ", data.dtype)


##### float64
fdata = data.get_fdata()
print("fdata: ", fdata.shape)
print("fdata: ", fdata.dtype)
# fdata_np = np.array(fdata).astype(np.int16)


##### 2D version
fdata_2D = fdata[:,75,:]
print("fdata_2D: ", fdata_2D.shape)
print("fdata_2D: ", fdata_2D.dtype)
# print("fdata_2D: ", fdata_2D)

# save image as HR
save_path = path + "T1_fdata_2D" + ".nii.gz"
nib.save(nib.Nifti1Image(fdata_2D, None, header=data.header.copy()), save_path)

##### numpy array
fdata_np = np.array(fdata_2D)
print("fdata_np: ", fdata_np.shape)
print("fdata_np: ", fdata_np.dtype)

# fdata_np_sq = np.squeeze(fdata_np)
# print("fdata_np_sq: ", fdata_np_sq.shape)
# print("fdata_np_sq: ", fdata_np_sq.dtype)

HR_img = fdata_np
# HR_img = np.array(nib.load(img_path).get_fdata()).astype(np.float32)


##### fft
HR_img_fft = np.fft.fftn(HR_img)
print("HR_img_fft: ", HR_img_fft.shape)
print("HR_img_fft: ", HR_img_fft.dtype)

# save_path = path + "T1_HR_img_fft" + ".nii.gz"
# nib.save(nib.Nifti1Image(np.real(HR_img_fft), None, header=data.header.copy()), save_path)

HR_img_fft_real = np.real(HR_img_fft)
HR_img_fft_real = (HR_img_fft_real * 255 / np.max(HR_img_fft_real)).astype(np.uint8)
Image.fromarray(HR_img_fft_real).save('T1_HR_img_fft_real.png') 
# reloaded = np.array(Image.open('T1_fdata_2D.png'))
# print(fdata_2D)
# print('\n\n')
# print(reloaded)


##### shift
HR_img_fft_shift = np.fft.fftshift(HR_img_fft)
print("HR_img_fft_shift: ", HR_img_fft_shift.shape)
print("HR_img_fft_shift: ", HR_img_fft_shift.dtype)

HR_img_fft_shift_real = np.real(HR_img_fft_shift)
HR_img_fft_shift_real = (HR_img_fft_shift_real * 255 / np.max(HR_img_fft_shift_real)).astype(np.uint8)
Image.fromarray(HR_img_fft_shift_real).save('T1_HR_img_fft_shift_real.png')


##### crop outside and pad with zeros
zeros_img = np.zeros((fdata_2D.shape[0], fdata_2D.shape[1]), dtype=np.complex128)
dim0_start = int(np.floor(fdata_2D.shape[0] / 4))
dim0_end = int(np.floor(fdata_2D.shape[0] * 3 / 4))
dim1_start = int(np.floor(fdata_2D.shape[1] / 4))
dim1_end = int(np.floor(fdata_2D.shape[1] * 3 / 4))
# print(dim0_start, dim0_end, dim1_start, dim1_end)
zeros_img[dim0_start:dim0_end,dim1_start:dim1_end] = HR_img_fft_shift[dim0_start:dim0_end,dim1_start:dim1_end]
LR_img_fft_shift_zero_padded = zeros_img
print("LR_img_fft_shift_zero_padded: ", LR_img_fft_shift_zero_padded.shape)
print("LR_img_fft_shift_zero_padded: ", LR_img_fft_shift_zero_padded.dtype)

LR_img_fft_shift_zero_padded_real = np.real(LR_img_fft_shift_zero_padded)
LR_img_fft_shift_zero_padded_real = (LR_img_fft_shift_zero_padded_real * 255 / np.max(LR_img_fft_shift_zero_padded_real)).astype(np.uint8)
Image.fromarray(LR_img_fft_shift_zero_padded_real).save('T1_LR_img_fft_shift_zero_padded_real.png') 


##### inverse shift
LR_img_fft = np.fft.ifftshift(LR_img_fft_shift_zero_padded)
print("LR_img_fft: ", LR_img_fft.shape)
print("LR_img_fft: ", LR_img_fft.dtype)

LR_img_fft_real = np.real(LR_img_fft)
LR_img_fft_real = (LR_img_fft_real * 255 / np.max(LR_img_fft_real)).astype(np.uint8)
Image.fromarray(LR_img_fft_real).save('T1_LR_img_fft_real.png')


##### inverse fft
LR_img = np.fft.ifftn(LR_img_fft)
print("LR_img: ", LR_img.shape)
print("LR_img: ", LR_img.dtype)

LR_img_real = np.real(LR_img)
print("LR_img_real: ", LR_img_real.shape)
print("LR_img_real: ", LR_img_real.dtype)

# save_path = path + "T1_HR_img_fft" + ".nii.gz"
# nib.save(nib.Nifti1Image(np.real(HR_img_fft), None, header=data.header.copy()), save_path)

# save image as LR
save_path = path + "T1_LR" + ".nii.gz"
nib.save(nib.Nifti1Image(LR_img_real, None, header=data.header.copy()), save_path)