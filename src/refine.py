import numpy as np
import bart, cfl
import os, time
import sigpy as sp
from sigpy import backend, util, config
from sigpy.alg import ConjugateGradient
from src.calibrate import *
from icecream import ic

import cupy as cp

from src.optimal_thresh import optht
from skimage.util import pad, view_as_windows
from scipy.linalg import null_space, svd

def VCC(kspace, caxis=0):
    vcc_ksp = kspace.copy()
    vcc_ksp = np.conj(vcc_ksp[:,::-1,::-1])
    if vcc_ksp.shape[1] % 2 == 0:
        vcc_ksp = np.roll(vcc_ksp,1,1)
    if vcc_ksp.shape[2] % 2 == 0:
        vcc_ksp = np.roll(vcc_ksp,1,2)
    out = np.concatenate((kspace,vcc_ksp), axis=caxis)
    return out

def dat2AtA(data, kernel_size):
    '''Computes the calibration matrix from calibration data.
    '''

    tmp = im2row(data, kernel_size)
    tsx, tsy, tsz = tmp.shape[:]
    A = np.reshape(tmp, (tsx, tsy*tsz), order='F')
    return np.dot(A.T.conj(), A)

def im2row(im, win_shape):
    '''res = im2row(im, winSize)'''
    sx, sy, sz = im.shape[:]
    wx, wy = win_shape[:]
    sh = (sx-wx+1)*(sy-wy+1)
    res = np.zeros((sh, wx*wy, sz), dtype=im.dtype)

    count = 0
    for y in range(wy):
        for x in range(wx):
            # res[:, count, :] = np.reshape(
            #     im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz), order='F')
            res[:, count, :] = np.reshape(
                im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz))
            count += 1
    return res


def calibrate_single_coil(AtA, kernel_size, ncoils, coil, lamda, sampling=None):

    kx, ky = kernel_size[:]
    if sampling is None:
        sampling = np.ones((*kernel_size, ncoils))
    dummyK = np.zeros((kx, ky, ncoils))
    dummyK[int(kx/2), int(ky/2), coil] = 1

    idxY = np.where(dummyK)
    idxY_flat = np.sort(
        np.ravel_multi_index(idxY, dummyK.shape, order='F'))
    sampling[idxY] = 0
    idxA = np.where(sampling)
    idxA_flat = np.sort(
        np.ravel_multi_index(idxA, sampling.shape, order='F'))

    Aty = AtA[:, idxY_flat]
    Aty = Aty[idxA_flat]

    AtA0 = AtA[idxA_flat, :]
    AtA0 = AtA0[:, idxA_flat]

    kernel = np.zeros(sampling.size, dtype=AtA0.dtype)
    lamda = np.linalg.norm(AtA0)/AtA0.shape[0]*lamda
    rawkernel = np.linalg.solve(AtA0 + np.eye(AtA0.shape[0])*lamda, Aty) # fast 1s

    kernel[idxA_flat] = rawkernel.squeeze()
    kernel = np.reshape(kernel, sampling.shape, order='F')

    return(kernel, rawkernel)


def spirit_calibrate(acs, kSize, lamda=0.001, filtering=True, verbose=True): # lamda=0.01
    nCoil = acs.shape[-1]
    AtA = dat2AtA(acs,kSize)
    if filtering: # singular value threshing
        if verbose:
            ic('prefiltering w/ opth')
        U,s,Vh = svd(AtA, full_matrices=False)
        k = optht(AtA, sv=s, sigma=None)
        if verbose:
            print('{}/{} kernels used'.format(k, len(s)))
        AtA= (U[:, :k] * s[:k] ).dot( Vh[:k,:])
        
    spirit_kernel = np.zeros((nCoil,nCoil,*kSize),dtype='complex128')
    for c in tqdm(range(nCoil)):
        tmp, _ = calibrate_single_coil(AtA,kernel_size=kSize,ncoils=nCoil,coil=c,lamda=lamda)
        spirit_kernel[c] = np.transpose(tmp,[2,0,1])
    spirit_kernel = np.transpose(spirit_kernel,[2,3,1,0]) # Now same as matlab!
    GOP = np.transpose(spirit_kernel[::-1,::-1],[3,2,0,1])
    GOP = GOP.copy()
    for n in range(nCoil):
        GOP[n,n,kSize[0]//2,kSize[1]//2] = -1  
    return GOP


def pruno_calibrate(acs,kSize, rank=300, verbose=True):
    nCoil = acs.shape[-1]
    kx, ky = kSize
    C = view_as_windows(acs, (kx,ky,nCoil)).reshape(-1,kx*ky*nCoil)
    u,s,vh = svd(C, full_matrices=True)
    pruno_kernel = vh[-rank:,:].T.conj()
#    pruno_kernel = null_space(C, rcond=1e-2)
    if verbose:
        print('original C', C.shape, 'pruno_kernel shape',pruno_kernel.shape)
    new_rank = pruno_kernel.shape[1]
    pruno_kernel = np.reshape(pruno_kernel, (kx, ky, nCoil, new_rank))
    pruno_kernel = np.transpose(pruno_kernel[::-1,::-1], [3,2,0,1])
    return pruno_kernel

class SPIRITHSPIRIT():
    """This performs F (G G^H)F^H x, where G is null-space calibrated from SPIRIT
        sp = SPIRITHSPIRIT(kernel) # kernel: [nCoil, nCoils, nX, nY]
    Args:
        x ([nCoil, nX, nY]): [kspace]
        null_kernel ([nCoil, nCoil, nX, nY]): [Calibrated SPIRIT image kernel] 

    Returns:
        [nCoil nX, nY]: [Gradient of the null operator]
    """
    def __init__(self, null_kernel, domain='image'):
        self.null_kernel = null_kernel
        self.domain = domain        
        
    def forward(self,x):
        xp = backend.get_array_module(x)
        im = sp.ifft(x, axes=(1,2)) # [nCoil, nX, nY]
        null_proj = xp.sum(im[None] * self.null_kernel, axis=1) - im # [nmaps, nX, nY]
        null_proj_kspace = sp.fft(null_proj, axes=(1,2))
        return null_proj_kspace
    def normal(self, x):
        xp = backend.get_array_module(x)
        if self.domain == 'image':
            im = sp.ifft(x, axes=(1,2)) # [nCoil, nX, nY]
            null_proj = xp.sum(im[None] * self.null_kernel, axis=1) # [nmaps, nX, nY]
            norm_proj = xp.sum(null_proj[:,None] * xp.conj(self.null_kernel), axis=0)
            norm_proj_kspace = sp.fft(norm_proj, axes=(1,2))
        else: # This is performing convolution need to check on this
            GOP_op = sp.linop.ConvolveData(data_shape = x.shape, filt = self.null_kernel, multi_channel=True, mode='full')
            norm_proj_kspace = GOP_op.H(GOP_op(x))

        return norm_proj_kspace
    
    
def vcc_calib(ksp, nacs):
    REG=nacs
    LOW=20
    
    tmp1 = bart.bart(1, 'flip 7', ksp)
    tmp2 = bart.bart(1, 'circshift 0 1', tmp1)
    tmp1 = bart.bart(1, 'circshift 1 1', tmp2)
    tmp2 = bart.bart(1, 'circshift 2 1', tmp1)
    vcs = bart.bart(1, 'conj', tmp2)
    
    #calibration
    both = bart.bart(1,'join 3', ksp, vcs)
    bsens = bart.bart(1,f'ecalib -r{REG} -a', both)
    COILS = bsens.shape[3]
#    COILS = bart.bart(1,'show -d3', bsens)
    HCOILS = COILS // 2
    maps1_coo = bart.bart(1, f'extract 3 0 {HCOILS}', bsens)
    maps2_coo = bart.bart(1, f'extract 3 {HCOILS} {COILS}', bsens)
    rel_coo = bart.bart(1, f'fmac -s8', maps1_coo, maps2_coo)
    ph_coo = bart.bart(1, 'cpyphs', rel_coo)
    phsqrt_coo = bart.bart(1,'spow 0.5', ph_coo)
    sens = bart.bart(1,'fmac -C', maps1_coo, phsqrt_coo)
    
    # extract low resolution phase
    sensLR = bart.bart(1,f'caldir {LOW}', ksp)
    ph = bart.bart(1,'cpyphs', sensLR)
    
    phi = bart.bart(1, 'scale -- 1.i', ph)
    phc = bart.bart(1, 'join 4', ph, phi)
    
    #align sign
    tmp1_coo = bart.bart(1,'fmac -s8 -C', sens, phc)
    tmp2_coo = bart.bart(1,'creal', tmp1_coo)
    tmp1_coo = bart.bart(1,'cpyphs', tmp2_coo)
    maps = bart.bart(1,'fmac', sens, tmp1_coo)
    maps = bart.bart(1, 'normalize 8', maps)

    return maps
    
class ESPHESP():
    """This performs F (S S^H - I)(S S^H -I)F^H x, where S is calibrated from ESPIRIT
        esp = ESPHESP(maps) # maps: [nCoil, nmaps, nX, nY]
    Args:
        x ([nCoil, nX, nY]): [kspace]
        null_kernel ([nCoil, nmaps, nX, nY]): [Calibrated Sensitivity maps, if two maps, 2nd dimension is 2] 

    Returns:
        [nCoil nX, nY]: [Gradient of the null operator]
    """
    def __init__(self, null_kernel):
        self.null_kernel = null_kernel
        
    def forward(self, x):        
        xp = backend.get_array_module(x)
        im = sp.ifft(x, axes=(1,2)) # [nCoil, nX, nY]
        ip = xp.sum(im[:,None] * xp.conj(self.null_kernel), axis=0) # [nmaps, nX, nY]
        proj = xp.sum(ip[None] * self.null_kernel, axis=1) # [nCoil, 1, nX, nY]
        null = im - proj
        return null
    def adjoint(self, x):     
        xp = backend.get_array_module(x)
        ip = xp.sum(x[:,None] * xp.conj(self.null_kernel), axis=0) # [nmaps, nX, nY]
        proj = xp.sum(ip[None] * self.null_kernel, axis=1) # [nCoil, 1, nX, nY]
        null = x - proj
        out = sp.fft(null, axes=(1,2))    
        return out 
    def normal(self, x):
        return self.adjoint(self.forward(x))


class RefineKspace():
    """[summary]

    Returns:
        [type]: [description]
    """
    
    def __init__(self, ksp_us, ksp_dl, nacs=24, kSize=[7,7],lam=10, rank=300, method='espirit', esp_map=None, domain='conv', autoespirit=False, vcc=False, verbose=True, device = sp.Device(0)):
        self.device = device
        self.vcc = vcc
        t0 = time.time() 
        
        if self.vcc:
            ksp_us = VCC(ksp_us)
            ksp_dl = VCC(ksp_dl)
            if nacs % 2 == 0:
                nacs = nacs - 1

        if method == 'espirit' and esp_map is None:
            if verbose:
                print('Calibrating ESPIRIT........')
                           
            ksp_for_bart = np.transpose(ksp_dl, (1,2,0))[None] 
            if vcc == True:
                sens_two_maps = vcc_calib(ksp_for_bart, nacs)
            else:
                if autoespirit:
                    sens_two_maps = bart.bart(1, f'ecalib -d 0 -r{nacs} -m 2 -a',ksp_for_bart)
                else:
                    sens_two_maps = bart.bart(1, f'ecalib -d 0 -r{nacs} -m 2',ksp_for_bart)
            sens_two_maps_reorder = np.transpose(sens_two_maps, (3,4,1,2,0))[...,0]
            s = sp.to_device(sens_two_maps_reorder, device=self.device)
            self.nullop = ESPHESP(s)

        if method == 'espirit' and esp_map is not None:
            if verbose:
                print('Using Calibrated espirit map........')
            s = sp.to_device(esp_map, device=self.device)
            self.nullop = ESPHESP(s)            
            
        elif method == 'spirit':
            if verbose:
                print('Calibrating SPIRIT........')            
            nCoil, nX, nY = ksp_us.shape
            acs = sp.resize(ksp_us, [nCoil, nX, nacs])
            acs = np.moveaxis(acs,0,-1)
            spirit_kernel = spirit_calibrate(acs,kSize=kSize, filtering=True, verbose=verbose)
            spirit_kernel = spirit_kernel.astype('complex64')
            if domain == 'image':
                spirit_img_kernel = sp.ifft(sp.resize(spirit_kernel, (nCoil,nCoil,nX,nY)), axes=(-1,-2)) * np.sqrt(nX * nY)
                s = sp.to_device(spirit_img_kernel, device=self.device)
                self.nullop = SPIRITHSPIRIT(s, domain=domain)
            else:
                s = spirit_kernel
                self.nullop = SPIRITHSPIRIT(s, domain=domain)
        elif method == 'pruno':
            if verbose:
                print('Calibrating Pruno........')            
            nCoil, nX, nY = ksp_us.shape
            acs = sp.resize(ksp_us, [nCoil, nX, nacs])
            acs = np.moveaxis(acs,0,-1)
            pruno_kernel = pruno_calibrate(acs,kSize=kSize, rank=rank, verbose=verbose)
            pruno_kernel = pruno_kernel.astype('complex64')
            self.nullop = SPIRITHSPIRIT(pruno_kernel, domain='conv')

        mask = np.abs(ksp_us) > 0
        mask = mask[:,0:1,:]
        mask = mask.astype('complex64')
        self.m = sp.to_device(mask, device=self.device)
        self.ksp_dl = sp.to_device(ksp_dl, device=self.device)
        self.ksp_us = sp.to_device(ksp_us, device=self.device)
        self.lam = lam
        self.verbose = verbose
        if verbose:
            print('Calibration Finished ...... {:.2f} seconds'.format(time.time() - t0)) 
            
    def GOPHGOP(self, x): 
        return self.nullop.normal(x)
    
    def Dc(self, x):
        return (1-self.m) * x
#        return x
    def D(self, x):
        return self.m * x 
    
    def A(self, x):
        data_consistency = self.lam * self.D(x)
        L2Reg = self.Dc(x)
        nullspace = self.lam * self.GOPHGOP(x)
        return data_consistency + L2Reg + nullspace        

    def run(self, niter=21, tol=1e-5):
        
        x0 = self.ksp_dl
        b = self.lam * self.ksp_us + self.Dc(self.ksp_dl)
        del self.ksp_us
        A = self.A
        alg = ConjugateGradient(A,b,x0,max_iter=niter, tol=tol)
        while not alg.done():
            alg.update()
            if self.verbose and alg.iter % 10 == 0:
                print(f'Iter [{alg.iter}/{niter}], residual:{alg.resid}')
        res = sp.to_device(alg.x, device=sp.cpu_device)
        if self.vcc:
            nCoil = len(res) // 2
            res = res[:nCoil]

        # Flush everything
        del self.m
        del self.ksp_dl
        del alg
        del b
        del self.nullop
        cp._default_memory_pool.free_all_blocks()

        return res
        
        
