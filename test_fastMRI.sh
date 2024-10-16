#!/bin/bash
# sh test_fastMRI.sh "vp"

export CUDA_VISIBLE_DEVICES=2

if [ "$1" = "vp" ]
then
    echo "================ run configs/vp/ddpm_continuous.py ================"
    python main.py \
        --config=configs/vp/ddpm_continuous.py \
        --mode='sample'  \
        --workdir=results
elif [ "$1" = "ve" ]
then
    echo "================ run configs/ve/ncsnpp_continuous.py ================"
    python main.py \
        --config=configs/ve/ncsnpp_continuous.py \
        --mode='sample'  \
        --workdir=results
elif [ "$1" = "spirit" ]
then
    echo "================ run configs/ve/ncsnpp_continuous.py ================"
    python main.py \
        --config=configs/SPIRiT/ncsnpp_continous.py \
        --mode='sample'  \
        --workdir=results
else
    echo "================ You must input one argument: ve or vp ================"
fi