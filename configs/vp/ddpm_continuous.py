"""Training DDPM with SPIRIT SDE."""
from configs.default_spirit_configs import get_default_configs


'''
  training TODO:
  training.sde, vp or ms
  training.mask_type 
  training.acs
  training.mean_equal
  training.acc
  sde的std M_hat
  beta_max, beta_min
  num_scales加速采样后续可能要改
  -------------------
  sampling TODO:
  training.sde, vp or spirit
  ---
  sampling.predictor
  sampling.corrector
  sampling.folder
  sampling.ckpt
  beta
  input initial
'''
def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.batch_size = 1
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama' # reverse_diffusion or euler_maruyama
  sampling.corrector = 'langevin' # langevin or none
  # sampling.folder = '2022_08_18T19_34_31_ddpm_vpsde_beta_20.0_N_1000'
  # sampling.folder = '2022_09_09T22_57_34_ddpm_spiritsde_VP-like_beta_200.0_N_1000'
  sampling.folder = '2022_10_21T15_38_07_ddpm_vpsde_alpha_20.0_N_1000'
  sampling.ckpt = 50
  sampling.mask_type = 'ellip' # uniform, random_uniform or center
  sampling.acc = '10'
  sampling.center = '48'
  sampling.snr = 0.16 ##### 0.16, 加速采样用0.26会比较好
  sampling.mse = 2.5 ##### predictor_mse
  sampling.corrector_mse = 5. ###

  # data
  data = config.data
  data.centered = False # True: Input is in [-1, 1]
  data.dataset_name = 'VWI' # fastMRI_knee or VWI
  data.image_size = 256
  data.normalize_type = 'std' # minmax or std
  data.normalize_coeff = 1.5 # normalize coefficient

  # model
  model = config.model
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  return config
