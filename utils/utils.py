import os
import torch
import numpy as np
import argparse
import torch.fft as FFT
import glob
import scipy.io as scio
import tensorflow as tf
import logging
import torch.nn.functional as F
import h5py
from pathlib import Path
from typing import Dict


def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_mat_np(save_dict, variable, file_name, index=0, normalize=True):
    if normalize:
        variable = normalize_complex(variable)
    file = os.path.join(save_dict, str(file_name) + "_" + str(index + 1) + ".mat")
    datadict = {str(file_name): np.squeeze(variable)}
    scio.savemat(file, datadict)


def save_mat(save_dict, variable, file_name, index=0, normalize=True):
    # variable = variable.cpu().detach().numpy()
    if normalize:

        variable = normalize_complex(variable)
    variable = variable.cpu().detach().numpy()
    file = os.path.join(save_dict, str(file_name) + "_" + str(index + 1) + ".mat")
    datadict = {str(file_name): np.squeeze(variable)}
    scio.savemat(file, datadict)


def get_all_files(folder, pattern="*"):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


"""
input:
    z: 18x2x256x256, [coil, channel, h, w]
    kernel: 18x18x7x7, [channel, coil, h, w]
"""


def kernel_to_matrix(z, kernel):
    w = torch.zeros(18, 18, 256, 256, dtype=torch.complex128)
    w[:, :, 124:131, 124:131] = kernel[:, :, :, :]
    w = torch.flip(w, dims=[-2, -1])
    w = torch.permute(
        w, (1, 0, 2, 3)
    )  #  [channel, coil, H, W] ===> [coil, channel, H, W]
    w = ifft2c_2d(w).to(z.device)  # 18x18x256x256, [coil, channel, H, W]

    res_kernel = torch.norm(Phi(fft2c_2d(r2c(z)), kernel))  # [coil, channel, H, W]
    res_matrix = torch.norm(torch.sum(fft2c_2d(w * r2c(z)), 0))
    conv_coeff = res_kernel / res_matrix

    return w, conv_coeff


def kernel_to_matrix_transpose(z, kernel):
    kernel = torch.permute(kernel, (0, 2, 1, 3, 4))
    kernel = torch.flip(kernel, dims=[-2, -1])
    kernel = torch.conj(kernel)

    w = torch.zeros(18, 18, 256, 256, dtype=torch.complex128)
    w[:, :, 124:131, 124:131] = kernel[:, :, :, :]
    w = torch.flip(w, dims=[-2, -1])
    w = torch.permute(
        w, (1, 0, 2, 3)
    )  #  [channel, coil, H, W] ===> [coil, channel, H, W]
    w = ifft2c_2d(w).to(z.device)  # 18x18x256x256, [coil, channel, H, W]

    res_kernel = torch.norm(Phi(fft2c_2d(r2c(z)), kernel))  # [coil, channel, H, W]
    res_matrix = torch.norm(torch.sum(fft2c_2d(w * r2c(z)), 0))
    conv_coeff = res_kernel / res_matrix

    return w, conv_coeff


def Phi(ksp, kernel):  # ksp: 18x1x256x302
    ksp = torch.permute(ksp, (1, 0, 2, 3))
    nb, nc, nc, kx, ky = kernel.size()  # 1x18x18x7x7
    res = torch.cat([complex_conv(ksp[i : i + 1], kernel[i]) for i in range(nb)], 0)
    res = torch.permute(res, (1, 0, 2, 3))
    # return r2c(res) # res is in kspace domain
    return res


def Phi_H(ksp, kernel):  # kspace: 18x1x256x256
    ksp = torch.permute(ksp, (1, 0, 2, 3))
    nb, nc, nc, kx, ky = kernel.size()  # 1x18x18x7x7

    filter_ = torch.permute(kernel, (0, 2, 1, 3, 4))
    filter = torch.flip(filter_, dims=[-2, -1])
    filter = torch.conj(filter)

    res = torch.cat([complex_conv(ksp[i : i + 1], filter[i]) for i in range(nb)], 0)
    res = torch.permute(res, (1, 0, 2, 3))
    # return r2c(res) # res is in kspace domain
    return res


def Phi_Phi_H(ksp, kernel):
    ksp = Phi(ksp, kernel)
    ksp = Phi_H(ksp, kernel)
    return ksp


def Psi(x, kernel):
    Psi_x = fft2c_2d(r2c(x))
    Psi_x = Psi_x - Phi(Psi_x, kernel) - Phi_H(Psi_x, kernel) + Phi_Phi_H(Psi_x, kernel)
    Psi_x = ifft2c_2d(Psi_x)
    Psi_x = c2r(Psi_x).type(torch.FloatTensor).to(x.device)
    return Psi_x


def orthogonal_csm(csm, z):
    csm = torch.permute(csm, (1, 0, 2, 3))
    Q_z = torch.conj(csm) * r2c(z)
    Q_z = torch.sum(Q_z, 0)
    Q_z = torch.unsqueeze(Q_z, 0)
    Q_z = Q_z * csm
    Q_z = c2r(Q_z)

    return Q_z


def complex_conv(x, kernel):  # x: 1x18x256x256
    nc, nc, kx, ky = kernel.size()  # 18x18x7x7

    x_real = torch.real(x).type(torch.FloatTensor).to(x.device)
    x_img = torch.imag(x).type(torch.FloatTensor).to(x.device)
    k_real = torch.real(kernel).type(torch.FloatTensor).to(x.device)
    k_img = torch.imag(kernel).type(torch.FloatTensor).to(x.device)
    padding = (kx // 2, ky // 2)
    res_real = F.conv2d(x_real, k_real, padding=padding) - F.conv2d(
        x_img, k_img, padding=padding
    )
    res_img = F.conv2d(x_real, k_img, padding=padding) + F.conv2d(
        x_img, k_real, padding=padding
    )
    # return torch.cat([res_real, res_img], 0)
    res = res_real + 1j * res_img
    return res


def readcfl(name):
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline()  # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split()]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[: np.searchsorted(dims_prod, n) + 1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    return a.reshape(dims, order="F")  # column-major


def to_tensor(x):
    re = np.real(x)
    im = np.imag(x)
    x = np.concatenate([re, im], 1)
    del re, im
    return torch.from_numpy(x)


def crop(img, cropx, cropy):
    nb, c, y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[:, :, starty : starty + cropy, startx : startx + cropx]


def normalize(img):
    """Normalize img in arbitrary range to [0, 1]"""
    img -= torch.min(img)
    img /= torch.max(img)
    return img


def normalize_np(img):
    """Normalize img in arbitrary range to [0, 1]"""
    img -= np.min(img)
    img /= np.max(img)
    return img


def normalize_complex(img):
    """normalizes the magnitude of complex-valued image to range [0, 1]"""
    abs_img = normalize(torch.abs(img))
    ang_img = normalize(torch.angle(img))
    return abs_img * torch.exp(1j * ang_img)


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def get_mask(config, caller):
    if caller == "sde":
        if config.training.mask_type == "low_frequency":
            mask_file = (
                "mask/"
                + config.training.mask_type
                + "_acs"
                + config.training.acs
                + ".mat"
            )
        elif config.training.mask_type == "center":
            mask_file = "mask/" + config.training.mask_type + "_acc2.mat"
        else:
            mask_file = (
                "mask/"
                + config.training.mask_type
                + "_acc"
                + config.training.acc
                + "_acs"
                + config.training.acs
                + ".mat"
            )
    elif caller == "sample":
        if config.sampling.mode == "fastMRI":
            mask_file = (
                "mask/"
                + config.sampling.mask_type
                + "_acc"
                + config.sampling.acc
                + "_acs"
                + config.sampling.center
                + ".mat"
            )
        else:
            mask_file = (
                "mask/"
                + config.sampling.mask_type
                + "_acc"
                + config.sampling.acc
                + "_center"
                + config.sampling.center
                + ".mat"
            )

    mask = scio.loadmat(mask_file)["mask"]
    mask = mask.astype(np.complex128)
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    mask = torch.from_numpy(mask).to(config.device)

    return mask


def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)


def fft2c(x):
    device = x.device
    nb, nc, nt, nx, ny = x.size()
    ny = torch.Tensor([ny]).to(device)
    nx = torch.Tensor([nx]).to(device)
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.fft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=4), torch.sqrt(ny))
    return x


def fft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny]).to(device)
    nx = torch.Tensor([nx]).to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.fft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.div(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(ny))
    return x


def FFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2) / np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.fft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3) / np.math.sqrt(ny)
    return x


def ifft2c(x):
    device = x.device
    nb, nc, nt, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.ifft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=4), torch.sqrt(ny))
    return x


def ifft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.ifft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.mul(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(ny))
    return x


def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.ifft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2) * np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.ifft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3) * np.math.sqrt(ny)
    return x


def IFFTc(x, axis, norm="ortho"):
    """expect x as m*n matrix"""
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis
    )


def Emat_xyt(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = r2c(b) * mask
            if b.ndim == 4:
                b = ifft2c_2d(b)
            else:
                b = ifft2c(b)
            x = c2r(b)
        else:
            b = r2c(b)
            if b.ndim == 4:
                b = fft2c_2d(b) * mask
            else:
                b = fft2c(b) * mask
            x = c2r(b)
    else:
        if inv:
            csm = r2c(csm)
            x = r2c(b) * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c(x)
            x = x * torch.conj(csm)
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)
            x = c2r(x)

        else:
            csm = r2c(csm)
            b = r2c(b)
            b = b * csm
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask * b
            x = c2r(x)

    return x


def Emat_xyt_complex(b, inv, csm, mask):
    if csm == None:  # 18x1x256x256
        if inv:
            b = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(b)
            else:
                x = ifft2c(b)
        else:
            if b.ndim == 4:
                x = fft2c_2d(b) * mask
            else:
                x = fft2c(b) * mask
    else:
        if inv:
            x = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c(x)
            x = x * torch.conj(csm)
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)

        else:
            b = b * csm
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask * b

    return x


def r2c(x):
    re, im = torch.chunk(x, 2, 1)
    x = torch.complex(re, im)
    return x


def c2r(x):
    x = torch.cat([torch.real(x), torch.imag(x)], 1)
    return x


def sos(x, dim):
    # x = r2c(x)
    x = torch.pow(torch.abs(x), 2)
    x = torch.sum(x, dim)
    x = torch.pow(x, 0.5)
    x = torch.unsqueeze(x, dim)
    return x


def Abs(x):
    x = r2c(x)
    return torch.abs(x)


def l2mean(x):
    result = torch.mean(torch.pow(torch.abs(x), 2))

    return result


def TV(x, norm="L1"):
    nb, nc, nx, ny = x.size()
    Dx = torch.cat([x[:, :, 1:nx, :], x[:, :, 0:1, :]], 2)
    Dy = torch.cat([x[:, :, :, 1:ny], x[:, :, :, 0:1]], 3)
    Dx = Dx - x
    Dy = Dy - x
    tv = 0
    if norm == "L1":
        tv = torch.mean(torch.abs(Dx)) + torch.mean(torch.abs(Dy))
    elif norm == "L2":
        Dx = Dx * Dx
        Dy = Dy * Dy
        tv = torch.mean(Dx) + torch.mean(Dy)
    return tv


def restore_checkpoint(ckpt_dir, state, device):

    loaded_state = torch.load(ckpt_dir, map_location=device)
    state["optimizer"].load_state_dict(loaded_state["optimizer"])
    state["model"].load_state_dict(loaded_state["model"], strict=False)
    state["ema"].load_state_dict(loaded_state["ema"])
    state["step"] = loaded_state["step"]

    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(saved_state, ckpt_dir)


def save_reconstructions(reconstructions: Dict[str, np.ndarray], out_dir: Path):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, "w") as hf:
            hf.create_dataset("reconstruction", data=recons)
