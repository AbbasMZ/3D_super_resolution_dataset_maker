# 3D_super_resolution_dataset_maker
This program generates low resolution (LR) and high resolution (HR) image pairs from a given dataset containing 3D nifti images.
Note that technically, the notion of LR here refers to downsampling as it fits medical imaging applications.

## Installation
Clone this repository:
```
git clone https://github.com/AbbasMZ/3D_super_resolution_dataset_maker.git
```
Either use the current environment or create a new virtual environment.
Make sure it contains python >= 3.5.
Then, install using pip:
```
cd 3D_super_resolution_dataset_maker
pip3 install -e .
```

## Usage
Change directory to `3d_upsampling_dataset_maker`.
Copy your 3D nifti dataset to a directory under the current path.

For 2D and 3D undersampling use `main_2D.py` and `main_3D.py` respectively.

You can set the downsampling factor `-f` for integer numbers and for example if you set it to 4, 1/4 of high frequency information will be removed and replaced with zero in the Fourier Space.
Using `-d` You can also choose to have the output of each file in a 3D format or a series of 2D files.
If you choose the output to be in 2D files, using `-x` being 0, 1, or 2, you can choose which axis to be used for the slicing.
In the case of brain MRI images, usually `-x 1` would result in axial slices.
