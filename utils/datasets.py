import os
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.utils import *


class ExampleDataSet(Dataset):
    def __init__(self, config):
        super(ExampleDataSet, self).__init__()
        self.config = config
        self.kspace_dir = "example_data/ksp.h5"
        self.maps_dir = "example_data/map_sos.h5"
        self.kernel_dir = "example_data/kernel_7x7_lamda_0p16.h5"

    def __getitem__(self, idx):

        with h5py.File(self.maps_dir, "r") as data:
            maps_idx = data["s_maps"][0]
            maps = np.asarray(maps_idx)

        with h5py.File(self.kernel_dir, "r") as data:
            kernel_idx = data["kernel"][0]
            kernel = np.asarray(kernel_idx)

        with h5py.File(self.kspace_dir, "r") as data:
            ksp_idx = data["kspace"][0]
            if self.config.data.normalize_type == "minmax":
                ksp_idx = torch.from_numpy(ksp_idx)
                maps = torch.from_numpy(maps)
                ksp_idx = torch.unsqueeze(ksp_idx, 0)
                maps = torch.unsqueeze(maps, 0)
                img_idx = Emat_xyt_complex(ksp_idx, True, maps, 1)
                img_idx = normalize_complex(img_idx)
                ksp_idx = Emat_xyt_complex(img_idx, False, maps, 1)
                ksp_idx = torch.squeeze(ksp_idx, 0)
                maps = torch.squeeze(maps, 0)
            elif self.config.data.normalize_type == "std":
                minv = np.std(ksp_idx)
                ksp_idx = ksp_idx / (self.config.data.normalize_coeff * minv)

            kspace = np.asarray(ksp_idx)

        return kspace, maps, kernel

    def __len__(self):
        # Total number of slices from all scans
        return 1


def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset_name)

    if config.data.dataset_name == "example":
        dataset = ExampleDataSet(config)
    if mode == "train":
        data = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            num_workers=6,
            shuffle=True,
            pin_memory=True,
        )
    else:
        data = DataLoader(
            dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    print(mode, "data loaded")

    return data
