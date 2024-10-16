
export CUDA_VISIBLE_DEVICES=2
mul=10
max=200
first=10.00
for i in `seq 101 $max`
do
    temp=`echo "scale=2; $i/$mul" | bc`
    echo $temp
    sed -i "s/sampling.mse = $first/sampling.mse = $temp/g" configs/ve/ncsnpp_continuous.py
    first=$temp
    python main.py \
        --config=configs/ve/ncsnpp_continuous.py \
        --mode='sample'  \
        --workdir=results
done