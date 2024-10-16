import os
import glob
import numpy as np
import h5py
import sigpy as sp
from utils.utils import *
import scipy.io as scio
import torch

output_dir = "example_data/map_sos.h5"
input_dir = "example_data/ksp.h5"


def main(input_dir, output_dir):

    mask = scio.loadmat("mask/center_acc48.mat")["mask"]
    mask = mask.astype(np.complex128)
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    mask = torch.from_numpy(mask)
    with h5py.File(input_dir, "r") as data:
        kspace = np.array(data["kspace"])  # 40x15x640x368
        print("kspace shape:", kspace.shape)

        kspace = torch.from_numpy(kspace)
        kspace = kspace * mask
        label_coil = ifft2c_2d(kspace)
        print(label_coil.shape)
        label = sos(label_coil, dim=1)
        s_maps = label_coil / label

        h5 = h5py.File(output_dir, "w")
        h5.create_dataset("s_maps", data=s_maps)
        h5.close()


if __name__ == "__main__":
    main(input_dir, output_dir)
