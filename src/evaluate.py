import skimage
import numpy as np
import cv2
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity



def metrics(est, gt):
    # psnr
    psnr = peak_signal_noise_ratio(
        gt,
        est,
        data_range=np.max(gt) - np.min(est)
    )

    # nrmse
    nrmse = normalized_root_mse(gt, est)

    # ssim
    # normalize 0 to 1

    ssim = structural_similarity(gt, est, data_range=gt.max())

    return ssim, psnr, nrmse


def laplacian_of_gaussian_2d(window_size, sigma):
    # 2d gaussian
    log = np.zeros((window_size, window_size))
    sd = sigma * sigma
    for x in range(window_size):
        for y in range(window_size):
            x_sq = (x - window_size//2)**2 + (y - window_size//2)**2
            log[x, y] = (x_sq / (2*sd) - 1) / \
                (np.pi * sd**2) * np.exp(-x_sq/(2*sd))

    return log

# cal hfen


def compare_hfen(ori, rec):
    ori = ori / 10
    rec = rec / 10
    operation = laplacian_of_gaussian_2d(15, 1.5)
    ori = cv2.filter2D(ori.astype('float32'), -1, operation,
                       borderType=cv2.BORDER_CONSTANT)
    rec = cv2.filter2D(rec.astype('float32'), -1, operation,
                       borderType=cv2.BORDER_CONSTANT)
    hfen = np.linalg.norm(ori-rec, ord='fro')
    return hfen


def gmsd(vref, vcmp, rescale=True, returnMap=False):
    """
    Compute Gradient Magnitude Similarity Deviation (GMSD) IQA metric
    :cite:`xue-2014-gradient`. This implementation is a translation of the
    reference Matlab implementation provided by the authors of
    :cite:`xue-2014-gradient`.
    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image
    rescale : bool, optional (default True)
      Rescale inputs so that `vref` has a maximum value of 255, as assumed
      by reference implementation
    returnMap : bool, optional (default False)
      Flag indicating whether quality map should be returned in addition to
      scalar score
    Returns
    -------
    score : float
      GMSD IQA metric
    quality_map : ndarray
      Quality map
    """

    # Input images in reference code on which this implementation is
    # based are assumed to be on range [0,...,255].
    if rescale:
        scl = (255.0 / 10)
    else:
        scl = np.float32(1.0)

    T = 170.0
    dwn = 2
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
    dy = dx.T

    ukrn = np.ones((2, 2)) / 4.0
    aveY1 = signal.convolve2d(scl * vref, ukrn, mode='same', boundary='symm')
    aveY2 = signal.convolve2d(scl * vcmp, ukrn, mode='same', boundary='symm')
    Y1 = aveY1[0::dwn, 0::dwn]
    Y2 = aveY2[0::dwn, 0::dwn]

    IxY1 = signal.convolve2d(Y1, dx, mode='same', boundary='symm')
    IyY1 = signal.convolve2d(Y1, dy, mode='same', boundary='symm')
    grdMap1 = np.sqrt(IxY1**2 + IyY1**2)

    IxY2 = signal.convolve2d(Y2, dx, mode='same', boundary='symm')
    IyY2 = signal.convolve2d(Y2, dy, mode='same', boundary='symm')
    grdMap2 = np.sqrt(IxY2**2 + IyY2**2)

    quality_map = (2*grdMap1*grdMap2 + T) / (grdMap1**2 + grdMap2**2 + T)
    score = np.std(quality_map)

    if returnMap:
        return (score, quality_map)
    else:
        return score
