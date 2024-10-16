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

# Lint as: python3
"""Training NCSN++ on Church with VE SDE."""

from configs.default_spirit_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    # TODO 1
    training.sde = "spiritsde"  # spiritsde
    # TODO 2
    training.csm = True
    training.estimate_csm = "bart"  # bart or sos
    training.continuous = True
    training.reduce_mean = True

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "reverse_diffusion"  # euler_maruyama or reverse_diffusion
    sampling.corrector = "langevin"
    # TODO 3
    # sampling.folder = "2022_12_15T11_14_43_ncsnpp_vesde_True_alpha_348_std_N_1000"
    sampling.folder = "2023_01_04T10_54_30_ncsnpp_spiritsde_False_alpha_348_std_N_1000"
    # sampling.folder = "2023_01_04T10_55_30_ncsnpp_vesde_False_alpha_348_std_N_1000"
    sampling.ckpt = 500
    # TODO 4
    sampling.mask_type = "ellip"  # ellip, uniform, random_uniform or center
    sampling.acc = "7.6"
    sampling.center = "48"
    # TODO 5
    sampling.mode = "retrospective"  # retrospective, prospective, fastMRI
    # TODO 6
    sampling.auto_tuning = False
    sampling.snr = 0.7
    sampling.mse = 0.2  ##### predictor_mse
    sampling.corrector_mse = 0.2  ###
    # sampling.corrector_mse = 5.0

    # data
    data = config.data
    data.centered = False
    data.dataset_name = "example"  # example VWI
    data.image_size = 320
    data.normalize_type = "std"  # minmax or std
    data.normalize_coeff = 1.5  # normalize coefficient

    # model
    model = config.model
    model.name = "ncsnpp"
    model.dropout = 0.0
    model.sigma_max = 348
    model.eta = 0.4  ###
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "output_skip"
    model.progressive_input = "input_skip"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.init_scale = 0.0
    model.fourier_scale = 16
    model.conv_size = 3

    return config
