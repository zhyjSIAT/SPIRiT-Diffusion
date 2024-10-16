# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.model_utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
import sde_lib
from models import model_utils as mutils
from utils.utils import *
from tqdm import trange, tqdm


_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == "ode":
        raise NotImplementedError
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == "pc":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(
            config=config,
            sde=sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=config.sampling.snr,
            corrector_mse=config.sampling.corrector_mse,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
        )
    elif sampler_name.lower() == "cg":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_mri_CG(
            sde=sde,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=config.sampling.snr,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
            config=config,
        )
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, config, sde, score_fn, probability_flow=False):
        super().__init__()
        self.config = config
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, config, sde, score_fn, snr, corrector_mse, n_steps):
        super().__init__()
        self.config = config
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.corrector_mse = corrector_mse
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, config, sde, score_fn, probability_flow=False):
        super().__init__(config, sde, score_fn, probability_flow)

    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        if isinstance(self.sde, sde_lib.SpiritSDE):
            x, x_mean = self.rsde.sde(x, t, atb, kernel, atb_mask, csm)
        else:
            dt = -1.0 / self.rsde.N
            z = torch.randn_like(x)
            drift, diffusion = self.rsde.sde(x, t, atb, atb_mask)
            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, config, sde, score_fn, probability_flow=False):
        super().__init__(config, sde, score_fn, probability_flow)

    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        if isinstance(self.sde, sde_lib.SpiritSDE):
            x, x_mean = self.rsde.discretize(x, t, atb, kernel, atb_mask, csm)
        else:
            f, G = self.rsde.discretize(x, t, atb, csm, atb_mask)
            z = torch.randn_like(x)
            x_mean = x - f
            x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, config, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        return x, x


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, config, sde, score_fn, snr, corrector_mse, n_steps):
        super().__init__(config, sde, score_fn, snr, corrector_mse, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
            and not isinstance(sde, sde_lib.SpiritSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        corrector_mse = self.corrector_mse

        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            if isinstance(sde, sde_lib.VESDE) and self.config.training.csm:
                meas_grad = Emat_xyt(x, False, c2r(csm), atb_mask) - c2r(atb)
                meas_grad = Emat_xyt(meas_grad, True, c2r(csm), atb_mask)
            else:
                meas_grad = Emat_xyt(x, False, None, atb_mask) - c2r(atb)
                meas_grad = Emat_xyt(meas_grad, True, None, atb_mask)
            grad = score_fn(x, t)
            meas_grad /= torch.norm(meas_grad)
            meas_grad *= torch.norm(grad)
            meas_grad *= corrector_mse

            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            x_mean = x + step_size[:, None, None, None] * (grad - meas_grad)
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

            x = x.type(torch.FloatTensor).to(x_mean.device)
            x_mean = x_mean.type(torch.FloatTensor).to(x.device)

        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, config, sde, score_fn, snr, corrector_mse, n_steps):
        pass

    def update_fn(self, x, t, atb, kernel, atb_mask, csm):
        return x, x


def shared_predictor_update_fn(
    x,
    t,
    atb,
    kernel,
    atb_mask,
    csm,
    config,
    sde,
    model,
    predictor,
    probability_flow,
    continuous,
):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(config, sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(config, sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, atb, kernel, atb_mask, csm)


def shared_corrector_update_fn(
    x,
    t,
    atb,
    kernel,
    atb_mask,
    csm,
    config,
    sde,
    model,
    corrector,
    continuous,
    snr,
    corrector_mse,
    n_steps,
):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(
            config, sde, score_fn, snr, corrector_mse, n_steps
        )
    else:
        corrector_obj = corrector(config, sde, score_fn, snr, corrector_mse, n_steps)
    return corrector_obj.update_fn(x, t, atb, kernel, atb_mask, csm)


def get_pc_sampler(
    config,
    sde,
    shape,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    corrector_mse,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        config=config,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        config=config,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        corrector_mse=corrector_mse,
        n_steps=n_steps,
    )

    def pc_sampler(model, atb, kernel, atb_mask, csm):
        """The PC sampler funciton.

        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)  # 1x2x256x256
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            # for i in range(sde.N):
            for i in trange(sde.N):
                t = timesteps[i]
                # print('====================', i)
                vec_t = torch.ones(1, device=t.device) * t
                x, x_mean = corrector_update_fn(
                    x, vec_t, atb, kernel, atb_mask, csm, model=model
                )
                x, x_mean = predictor_update_fn(
                    x, vec_t, atb, kernel, atb_mask, csm, model=model
                )
                x = x.type(torch.FloatTensor).to(device)
                x_mean = x_mean.type(torch.FloatTensor).to(device)

            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

    return pc_sampler


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    A^{T}A * X + \lamda *X
    """

    def __init__(self, csm, mask, lam):
        self.pixels = mask.shape[0] * mask.shape[1]
        self.mask = mask
        self.csm = csm
        self.SF = torch.complex(
            torch.sqrt(torch.tensor(self.pixels).float()), torch.tensor(0.0).float()
        )
        self.lam = lam

    def myAtA(self, img):
        x = Emat_xyt(img, False, self.csm, self.mask)
        x = Emat_xyt(x, True, self.csm, self.mask)
        return x + self.lam * img


def myCG(A, Rhs, x0, it):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """

    Rhs = r2c(Rhs) + A.lam * r2c(x0)

    x = r2c(x0)
    i = 0
    r = Rhs - r2c(A.myAtA(x0))
    p = r
    rTr = torch.sum(torch.conj(r) * r).float()

    while i < it:

        Ap = r2c(A.myAtA(c2r(p)))
        alpha = rTr / torch.sum(torch.conj(p) * Ap).float()
        alpha = torch.complex(alpha, torch.tensor(0.0).float().cuda())
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(torch.conj(r) * r).float()
        beta = rTrNew / rTr
        beta = torch.complex(beta, torch.tensor(0.0).float().cuda())
        p = r + beta * p
        i = i + 1
        rTr = rTrNew

    return c2r(x)


def get_pc_mri_CG(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
    save_progress=False,
    save_root=None,
    lamb_schedule=None,
    lamb_schedule_interp=None,
    mask=None,
    measurement_noise=False,
    config=None,
):
    sampling_shape = (1, 2, 320, 320)
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def get_update_fn(update_fn):

        def radon_update_fn(
            model,
            data,
            x,
            t,
            mask,
            csm,
            measurement=None,
            i=None,
            pCondition=False,
            kernel=False,
        ):
            with torch.no_grad():
                vec_t = torch.ones(sampling_shape[0], device=data.device) * t
                gt, gt_next, score = update_fn(
                    x,
                    vec_t,
                    atb=data,
                    kernel=kernel,
                    model=model,
                    atb_mask=mask,
                    config=config,
                    csm=csm,
                )

                x0 = x + gt**2 * score
                Aobj = Aclass(csm, mask, torch.tensor(1.50).cuda())
                Rhs = Emat_xyt(c2r(measurement), True, csm, 1)

                x0 = myCG(Aobj, Rhs, x0, 5)

                c1 = (1 - gt_next**2 / gt**2).sqrt()
                eta = 0.75  #### eta=0.5
                c2 = -gt_next * gt * (1 - c1**2 * eta**2).sqrt()
                c3 = eta * gt_next * c1
                x = x0 + c2 * score + c3 * torch.randn_like(x)

                return x.to(torch.float)

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(
            model,
            data,
            x,
            t,
            mask,
            csm,
            interp,
            measurement=None,
            i=None,
            ksp_init=None,
            kernel=None,
            end=0,
        ):
            vec_t = torch.ones(sampling_shape[0], device=data.device) * t

            # input to the score function
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            lamb = lamb_schedule.get_current_lambda(i)

            AH_bmAX_next = Emat_xyt(
                c2r(measurement) - Emat_xyt(x_next, False, csm, mask), True, csm, mask
            )
            x_next = x_next + lamb * AH_bmAX_next

            x_final = x_next

            x_next = x_next.detach()
            x_final = x_final.detach()
            return x_next.to(torch.float), x_final.to(torch.float)

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(
        model,
        data,
        kernel,
        mask,
        csm,
        measurement=None,
        pc=True,
    ):

        csm = c2r(csm)
        x = sde.prior_sampling(sampling_shape).to(data.device)
        timesteps = torch.linspace(sde.T, eps, sde.N)

        print(sampling_shape)
        measurement = data
        ksp_init = measurement

        start_i = 0  ## unifrom1d
        vec_t = torch.ones(sampling_shape[0], device=data.device) * timesteps[start_i]
        x_init = Emat_xyt(c2r(ksp_init), True, csm, 1)
        x_mean, std = sde.marginal_prob(x_init, vec_t)
        x = x_mean + torch.randn_like(x_mean) * std[:, None, None, None]
        x = x.to(torch.float)
        ####
        csm = r2c(csm)

        ones = torch.ones_like(x).to(data.device)

        for i in tqdm(range(start_i, sde.N)):
            t = timesteps[i]
            x = predictor_denoise_update_fn(
                model,
                data,
                x,
                t,
                mask,
                csm,
                measurement=measurement,
                i=i,
                pCondition=pc,
            )
            if i == sde.N - 1:
                x, x_final = corrector_radon_update_fn(
                    model,
                    data,
                    x,
                    t,
                    mask,
                    csm,
                    0,
                    measurement=measurement,
                    i=i,
                    ksp_init=ksp_init,
                    kernel=kernel,
                    end=1,
                )
            else:
                x, _ = corrector_radon_update_fn(
                    model,
                    data,
                    x,
                    t,
                    mask,
                    csm,
                    0,
                    measurement=measurement,
                    i=i,
                    ksp_init=ksp_init,
                    kernel=kernel,
                    end=0,
                )

        return inverse_scaler(x if denoise else x), inverse_scaler(
            x_final if denoise else x_final
        )

    return pc_radon
