import numpy as np
import sigpy as sp
from icecream import ic
from tqdm import tqdm
from scipy.linalg import svd
from src.optimal_thresh import optht


def dat2AtA(data, kernel_size):
    """Computes the calibration matrix from calibration data."""

    tmp = im2row(data, kernel_size)
    tsx, tsy, tsz = tmp.shape[:]
    A = np.reshape(tmp, (tsx, tsy * tsz), order="F")
    return np.dot(A.T.conj(), A)


def im2row(im, win_shape):
    """res = im2row(im, winSize)"""
    sx, sy, sz = im.shape[:]
    wx, wy = win_shape[:]
    sh = (sx - wx + 1) * (sy - wy + 1)
    res = np.zeros((sh, wx * wy, sz), dtype=im.dtype)

    count = 0
    for y in range(wy):
        for x in range(wx):

            res[:, count, :] = np.reshape(
                im[x : sx - wx + x + 1, y : sy - wy + y + 1, :], (sh, sz)
            )
            count += 1
    return res


def calibrate_single_coil(AtA, kernel_size, ncoils, coil, lamda, sampling=None):

    kx, ky = kernel_size[:]
    if sampling is None:
        sampling = np.ones((*kernel_size, ncoils))
    dummyK = np.zeros((kx, ky, ncoils))
    dummyK[int(kx / 2), int(ky / 2), coil] = 1

    idxY = np.where(dummyK)
    idxY_flat = np.sort(np.ravel_multi_index(idxY, dummyK.shape, order="F"))
    sampling[idxY] = 0
    idxA = np.where(sampling)
    idxA_flat = np.sort(np.ravel_multi_index(idxA, sampling.shape, order="F"))

    Aty = AtA[:, idxY_flat]
    Aty = Aty[idxA_flat]

    AtA0 = AtA[idxA_flat, :]
    AtA0 = AtA0[:, idxA_flat]

    kernel = np.zeros(sampling.size, dtype=AtA0.dtype)
    lamda = np.linalg.norm(AtA0) / AtA0.shape[0] * lamda
    rawkernel = np.linalg.solve(AtA0 + np.eye(AtA0.shape[0]) * lamda, Aty)  # fast 1s

    kernel[idxA_flat] = rawkernel.squeeze()
    kernel = np.reshape(kernel, sampling.shape, order="F")

    return (kernel, rawkernel)


def spirit_calibrate(
    acs, kSize, lamda=0.001, filtering=False, verbose=True
):  # lamda=0.01
    nCoil = acs.shape[-1]
    AtA = dat2AtA(acs, kSize)
    if filtering:  # singular value threshing
        if verbose:
            ic("prefiltering w/ opth")
        U, s, Vh = svd(AtA, full_matrices=False)
        k = optht(AtA, sv=s, sigma=None)
        if verbose:
            print("{}/{} kernels used".format(k, len(s)))
        AtA = (U[:, :k] * s[:k]).dot(Vh[:k, :])

    spirit_kernel = np.zeros((nCoil, nCoil, *kSize), dtype="complex128")
    for c in tqdm(range(nCoil)):
        tmp, _ = calibrate_single_coil(
            AtA, kernel_size=kSize, ncoils=nCoil, coil=c, lamda=lamda
        )
        spirit_kernel[c] = np.transpose(tmp, [2, 0, 1])
    spirit_kernel = np.transpose(spirit_kernel, [2, 3, 1, 0])  # Now same as matlab!
    GOP = np.transpose(spirit_kernel[::-1, ::-1], [3, 2, 0, 1])
    GOP = GOP.copy()
    for n in range(nCoil):
        GOP[n, n, kSize[0] // 2, kSize[1] // 2] = -1
    return spirit_kernel


class CalibrateRSC:
    """
    Usage:
    null_kernel = CalibrateRSC(ksp, method='spirit or espirit', verbose=True, save_img=True)

    Inputs:
    ksp : (nCoil nX, nY)

    Outputs:
    null_kernel : (nCoil, nCoil, nX, nY)

    # Null Projection
    1. For SPIRIT: proj_null = np.sum(img[None] * null_kernel, axis=1) - img
        # Where img = sp.ifft(ksp,axes=(-1,-2))

    2. For ESPIRiT:

    """

    def __init__(
        self,
        ksp,
        method="spirit",
        nacs=24,
        kSize=(5, 5),
        vcc=True,
        filtering=False,
        verbose=True,
    ):

        self.method = method
        if method == "spirit":
            ic("calibrate spirit kernels in img domain")
            nCoil, nX, nY = ksp.shape
            img = sp.ifft(ksp, axes=(-1, -2))

            acs = sp.resize(ksp, [nCoil, nX, nacs])
            acs = np.moveaxis(acs, 0, -1)

            spirit_kernel = spirit_calibrate(
                acs, kSize=kSize, filtering=filtering, verbose=verbose
            )

            self.img_kernel = sp.ifft(
                sp.resize(spirit_kernel, (nCoil, nCoil, nX, nY)), axes=(-1, -2)
            ) * np.sqrt(nX * nY)

    def forward(self, ksp, input_type="kspace", output_type="kspace"):
        if input_type == "kspace":
            img = sp.ifft(ksp, axes=(-1, -2))
        else:
            img = ksp  # input is image

        proj_img = np.sum(img[None] * self.img_kernel, axis=1) - img

        if output_type == "kspace":
            proj_img = sp.fft(proj_img, axes=(-1, -2))

        return proj_img

    def adjoint(
        self, ksp, input_type="kspace", output_type="kspace"
    ):  # Adjoint operator
        if input_type == "kspace":
            img = sp.ifft(ksp, axes=(-1, -2))
        else:
            img = ksp  # input is image

        proj_img = np.sum(img[:, None] * np.conj(self.img_kernel), axis=0) - img

        if output_type == "kspace":
            proj_img = sp.fft(proj_img, axes=(-1, -2))

        return proj_img

    def normal(self, ksp, input_type="kspace", output_type="kspace"):

        img = sp.ifft(ksp, axes=(-1, -2))
        null_proj = np.sum(img[None] * self.img_kernel, axis=1) - img

        norm_proj = (
            np.sum(null_proj[:, None] * np.conj(self.img_kernel), axis=0) - null_proj
        )

        return sp.fft(norm_proj, axes=(-1, -2))
