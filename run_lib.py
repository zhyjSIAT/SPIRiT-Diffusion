"""Training and evaluation for score-based generative models. """

import os
import time
import tensorflow as tf
import logging
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# Keep the import below for registering all model definitions
from models import ncsnpp, ddpm
import losses
import sampling
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from utils.utils import *
import utils.datasets as datasets

FLAGS = flags.FLAGS


def nmse(recon, label):
    nmse_value = (torch.norm(recon - label) ** 2) / (torch.norm(label) ** 2)
    return nmse_value.double()


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # The directory for saving test results during training
    sample_dir = os.path.join(workdir, "samples_in_train")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    tf.io.gfile.makedirs(checkpoint_dir)
    # Resume training when intermediate checkpoints are detected
    initial_step = int(state["step"])

    # Build pytorch dataloader for training
    train_dl = datasets.get_dataset(config, "train")

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(config)
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(config)
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
    elif config.training.sde.lower() == "spiritsde":
        sde = sde_lib.SpiritSDE(config)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        config,
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for epoch in range(config.training.epochs):
        loss_sum = 0
        for step, batch in enumerate(train_dl):

            t0 = time.time()
            """make sure that the size of k0, kernel and csm are following:"""
            # k0: (batch_size,coil_map,kx,ky)
            # kernel: (batch_size,coil_map,coil_map,kernel_size,kernel_size)
            # csm: (batch_size,coil_map,kx,ky)
            k0, csm, kernel = batch
            k0 = k0.to(config.device)
            csm = csm.to(config.device)
            kernel = kernel.to(config.device)

            if config.training.sde == "vesde" and config.training.csm:
                label = Emat_xyt_complex(k0, True, csm, 1)
            else:
                k0 = torch.permute(k0, (1, 0, 2, 3))
                label = Emat_xyt_complex(k0, True, None, 1.0)

            label = c2r(label).type(torch.FloatTensor).to(config.device)
            label = scaler(label)

            loss = train_step_fn(state, label, kernel, csm)
            loss_sum += loss

            param_num = sum(param.numel() for param in state["model"].parameters())
            if step % 10 == 0:
                print(
                    "Epoch",
                    epoch + 1,
                    "/",
                    config.training.epochs,
                    "Step",
                    step,
                    "loss = ",
                    loss.cpu().data.numpy(),
                    "loss mean =",
                    loss_sum.cpu().data.numpy() / (step + 1),
                    "time",
                    time.time() - t0,
                    "param_num",
                    param_num,
                )

            # Report the loss on an evaluation dataset periodically
            if step % config.training.eval_freq == 0:
                pass

        # Save a checkpoint for every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}.pth"), state
            )


def sample(config, workdir):
    """Generate samples.

    Args:
      config: Configuration to use.
      workdir: Working directory.
    """
    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{config.sampling.ckpt}.pth")
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    print("load weights:", ckpt_path)

    SAMPLING_FOLDER_ID = "_".join(
        [
            FLAGS.config.sampling.acc,
            FLAGS.config.sampling.mode,
            FLAGS.config.sampling.center,
            FLAGS.config.sampling.mask_type,
            "ckpt",
            str(config.sampling.ckpt),
            FLAGS.config.sampling.predictor,
            FLAGS.config.sampling.corrector,
            str(config.sampling.snr),
            FLAGS.config.training.sde,
            str(FLAGS.config.model.eta),
            str(FLAGS.config.sampling.mse),
            str(FLAGS.config.sampling.corrector_mse),
        ]
    )
    # Build data pipeline

    if config.sampling.mode == "example":
        test_dl = datasets.get_dataset(config, "example")
    else:
        test_dl = datasets.get_dataset(config, "test_t1_all")
    FLAGS.config.sampling.folder = os.path.join(
        FLAGS.workdir, config.training.estimate_csm + "_acc" + SAMPLING_FOLDER_ID
    )
    tf.io.gfile.makedirs(FLAGS.config.sampling.folder)

    # Create data scaler and its inverse
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == "spiritsde":
        sde = sde_lib.SpiritSDE(config)
        sampling_eps = 1e-5  # TODO
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    """Build the sampling function when sampling is enabled, number stands for the number of coil map"""
    if config.training.sde == "vesde" and config.training.csm:
        sampling_shape = (
            1,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
    elif config.sampling.mode == "fastMRI":
        sampling_shape = (
            15,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
    else:
        sampling_shape = (
            18,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
        )
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, inverse_scaler, sampling_eps
    )

    if config.sampling.mode != "prospective":
        print("============no prospective mask!!!============")
        mask = get_mask(config, "sample")

    f = open(os.path.join(workdir, "snr_results.txt"), "a")
    ssim = StructuralSimilarityIndexMeasure()
    psnr = PeakSignalNoiseRatio()

    for index, point in enumerate(test_dl):

        print("index:", index)
        k0, csm, kernel = point
        k0 = k0.to(config.device)
        kernel = kernel.to(config.device)
        csm = csm.to(config.device)

        k0 = torch.permute(k0, (1, 0, 2, 3))  # 18x1x320x320
        if config.sampling.mode == "prospective":
            print("============prospective mask!!!============")
            mask = k0[0, 0, :, :].clone()
            mask[mask != 0] = 1
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)

        if config.training.estimate_csm == "sos":
            label = Emat_xyt_complex(k0, True, None, 1.0).to(config.device)
            label = sos(label, dim=0).type(torch.FloatTensor)
        else:
            label = Emat_xyt_complex(k0.permute(1, 0, 2, 3), True, csm, 1.0).to(
                config.device
            )

        atb = k0 * mask
        if config.training.sde == "vesde" and config.training.csm:
            atb = torch.permute(atb, (1, 0, 2, 3))

        recon, n = sampling_fn(score_model, atb, kernel, mask, csm)

        recon = r2c(recon)
        if config.training.estimate_csm == "sos":
            recon = sos(recon, dim=0)
        else:
            if config.training.sde == "spiritsde":
                recon = fft2c_2d(recon)
                recon = Emat_xyt_complex(recon.permute(1, 0, 2, 3), True, csm, 1.0).to(
                    config.device
                )

        save_mat(FLAGS.config.sampling.folder, recon, "recon", index, normalize=False)

        SSIM = ssim(recon.type(torch.FloatTensor), label.type(torch.FloatTensor))
        PSNR = psnr(recon.type(torch.FloatTensor), label.type(torch.FloatTensor))
        NMSE = nmse(recon.type(torch.FloatTensor), label.type(torch.FloatTensor))
        print("PSNR:", PSNR, "SSIM:", SSIM, "NMSE:", NMSE)
        print(FLAGS.config.sampling.folder)
        f.write(
            "eta="
            + str(FLAGS.config.model.eta)
            + ", mse="
            + str(FLAGS.config.sampling.mse)
            + ", corrector_mse="
            + str(FLAGS.config.sampling.corrector_mse)
            + ", snr="
            + str(FLAGS.config.sampling.snr)
            + ": PSNR = "
            + str(PSNR)
            + ", SSIM = "
            + str(SSIM)
            + "\n"
        )

    f.write(
        "-----------------------------------------------------------------------------------\n"
    )
    f.close()
