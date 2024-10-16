from src.calibrate import *
import sigpy as sp
import h5py
import numpy as np

"""load kspace data"""
ksp_path = "/data0/yuewang/data/VWI_kspace_320/example_data/ksp.h5"


# make sure that ksp shape is (n_coil_map,kx,ky)
with h5py.File(ksp_path, "r") as data:
    ksp = data["kspace"][:]

# only remain low-frequency data
n = 48

_, _, h, w = ksp.shape
start_h = (h - n) // 2
start_w = (w - n) // 2

masked_ksp = np.zeros_like(ksp)
masked_ksp[:, :, start_h : start_h + n, start_w : start_w + n] = ksp[
    :, :, start_h : start_h + n, start_w : start_w + n
]

cailb = sp.resize(
    np.transpose(masked_ksp.squeeze(), (1, 2, 0)),
    (12, 12, masked_ksp.shape[1]),
)

"""estimate kernel"""
kernel = spirit_calibrate(cailb, (7, 7), lamda=0.16, filtering=False, verbose=True)
kernel = np.transpose(kernel, (3, 2, 0, 1))
kernel = kernel[None]
# save kernel
save_file = "/data0/yuewang/data/VWI_kspace_320/example_data/kernel_7x7_lamda_0p16.h5"
with h5py.File(save_file, "w") as data:
    data.create_dataset("kernel", data=kernel)
