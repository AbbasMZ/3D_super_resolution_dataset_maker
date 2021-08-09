from re import A, X
import numpy as np
import scipy.fft as spfft
import nibabel as nib
import torch.nn
import torch
from PIL import Image

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# img = mpimg.imread('sample.jpeg')

factor = 4
epsilon = 1e-14

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
# fdata_np = np.array(fdata_2D)
fdata_np = np.array(fdata)
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

# HR_img_fft_real = np.real(HR_img_fft)
HR_img_fft_real_2D = np.real(HR_img_fft[:,126,:])
HR_img_fft_real_2D = (HR_img_fft_real_2D * 255 / np.max(HR_img_fft_real_2D)).astype(np.uint8)
Image.fromarray(HR_img_fft_real_2D).save('T1_HR_img_fft_real_3D_2nd.png') 
# reloaded = np.array(Image.open('T1_fdata_2D.png'))
# print(fdata_2D)
# print('\n\n')
# print(reloaded)

HR_img_fft_real_2D = np.real(HR_img_fft[126,:,:])
HR_img_fft_real_2D = (HR_img_fft_real_2D * 255 / np.max(HR_img_fft_real_2D)).astype(np.uint8)
Image.fromarray(HR_img_fft_real_2D).save('T1_HR_img_fft_real_3D_1st.png') 

HR_img_fft_real_2D = np.real(HR_img_fft[:,:,75])
HR_img_fft_real_2D = (HR_img_fft_real_2D * 255 / np.max(HR_img_fft_real_2D)).astype(np.uint8)
Image.fromarray(HR_img_fft_real_2D).save('T1_HR_img_fft_real_3D_3rd.png') 

##### shift
HR_img_fft_shift = np.fft.fftshift(HR_img_fft)
print("HR_img_fft_shift: ", HR_img_fft_shift.shape)
print("HR_img_fft_shift: ", HR_img_fft_shift.dtype)

HR_img_fft_shift_real = np.real(HR_img_fft_shift[126,:,:])
HR_img_fft_shift_real = (HR_img_fft_shift_real * 255 / np.max(HR_img_fft_shift_real)).astype(np.uint8)
Image.fromarray(HR_img_fft_shift_real).save('T1_HR_img_fft_shift_real_3D_1st.png')

HR_img_fft_shift_real = np.real(HR_img_fft_shift[:,126,:])
HR_img_fft_shift_real = (HR_img_fft_shift_real * 255 / np.max(HR_img_fft_shift_real)).astype(np.uint8)
Image.fromarray(HR_img_fft_shift_real).save('T1_HR_img_fft_shift_real_3D_2nd.png')

HR_img_fft_shift_real = np.real(HR_img_fft_shift[:,:,75])
HR_img_fft_shift_real = (HR_img_fft_shift_real * 255 / np.max(HR_img_fft_shift_real)).astype(np.uint8)
Image.fromarray(HR_img_fft_shift_real).save('T1_HR_img_fft_shift_real_3D_3rd.png')

##### crop outside and pad with zeros
zeros_img = np.zeros((fdata.shape[0], fdata.shape[1], fdata.shape[2]), dtype=np.complex128)
dim0_start = int(np.floor(fdata.shape[0] * (factor - 1) / (2 * factor)))
dim0_end = int(np.floor(fdata.shape[0] * (factor + 1) / (2 * factor)))
dim1_start = int(np.floor(fdata.shape[1] * (factor - 1) / (2 * factor)))
dim1_end = int(np.floor(fdata.shape[1] * (factor + 1) / (2 * factor)))
dim2_start = int(np.floor(fdata.shape[2] * (factor - 1) / (2 * factor)))
dim2_end = int(np.floor(fdata.shape[2] * (factor + 1) / (2 * factor)))
print("dim0_start, dim0_end, dim1_start, dim1_end, dim2_start, dim2_end: ", dim0_start, dim0_end, dim1_start, dim1_end, dim2_start, dim2_end)
zeros_img[dim0_start:dim0_end, dim1_start:dim1_end, dim2_start:dim2_end] = HR_img_fft_shift[dim0_start:dim0_end, dim1_start:dim1_end, dim2_start:dim2_end]
LR_img_fft_shift_zero_padded = zeros_img
print("LR_img_fft_shift_zero_padded: ", LR_img_fft_shift_zero_padded.shape)
print("LR_img_fft_shift_zero_padded: ", LR_img_fft_shift_zero_padded.dtype)

LR_img_fft_shift_zero_padded_real = np.real(LR_img_fft_shift_zero_padded[126,:,:])
LR_img_fft_shift_zero_padded_real = (LR_img_fft_shift_zero_padded_real * 255 / np.max(LR_img_fft_shift_zero_padded_real)).astype(np.uint8)
Image.fromarray(LR_img_fft_shift_zero_padded_real).save('T1_LR_img_fft_shift_zero_padded_real_3D_1st.png') 

LR_img_fft_shift_zero_padded_real = np.real(LR_img_fft_shift_zero_padded[:,126,:])
LR_img_fft_shift_zero_padded_real = (LR_img_fft_shift_zero_padded_real * 255 / np.max(LR_img_fft_shift_zero_padded_real)).astype(np.uint8)
Image.fromarray(LR_img_fft_shift_zero_padded_real).save('T1_LR_img_fft_shift_zero_padded_real_3D_2nd.png') 

LR_img_fft_shift_zero_padded_real = np.real(LR_img_fft_shift_zero_padded[:,:,75])
LR_img_fft_shift_zero_padded_real = (LR_img_fft_shift_zero_padded_real * 255 / np.max(LR_img_fft_shift_zero_padded_real)).astype(np.uint8)
Image.fromarray(LR_img_fft_shift_zero_padded_real).save('T1_LR_img_fft_shift_zero_padded_real_3D_3rd.png') 

##### inverse shift
LR_img_fft = np.fft.ifftshift(LR_img_fft_shift_zero_padded)
print("LR_img_fft: ", LR_img_fft.shape)
print("LR_img_fft: ", LR_img_fft.dtype)

LR_img_fft_real = np.real(LR_img_fft[126,:,:])
LR_img_fft_real = ((LR_img_fft_real - np.min(LR_img_fft_real)) * 255 / (np.max((LR_img_fft_real - np.min(LR_img_fft_real)))+epsilon)).astype(np.uint8)
Image.fromarray(LR_img_fft_real).save('T1_LR_img_fft_real_3D_1st.png')

LR_img_fft_real = np.real(LR_img_fft[:,126,:])
LR_img_fft_real = ((LR_img_fft_real - np.min(LR_img_fft_real)) * 255 / (np.max((LR_img_fft_real - np.min(LR_img_fft_real)))+epsilon)).astype(np.uint8)
Image.fromarray(LR_img_fft_real).save('T1_LR_img_fft_real_3D_2nd.png')

LR_img_fft_real = np.real(LR_img_fft[:,:,75])
LR_img_fft_real = ((LR_img_fft_real - np.min(LR_img_fft_real)) * 255 / (np.max((LR_img_fft_real - np.min(LR_img_fft_real)))+epsilon)).astype(np.uint8)
Image.fromarray(LR_img_fft_real).save('T1_LR_img_fft_real_3D_3rd.png')

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
save_path = path + "T1_LR_factor_" + str(factor) + ".nii.gz"
print('save_path: ', save_path)
nib.save(nib.Nifti1Image(LR_img_real, None, header=data.header.copy()), save_path)