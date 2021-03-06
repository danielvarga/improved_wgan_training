# no real difference between baseline (65.71%), datagrad (65.76%) and spectreg (65.81%) when augmentation is used
NAME=cifar100_dg_spect_aug
mkdir -p couts/
echo $NAME

D=1
DATASET=cifar100
ITERS=50000
BS=128
LR=0.01
LRD=piecewise
WD=0.003
NET=cifarResnet
BN=True
TRAIN=20000
AUGMENTATION=True
WIDENESS=3
DG=0.003
LAMBDA=0.00003

for C in `seq 1 1 10`
do
    echo ITER $C
    COMMON_ARGS="--RANDOM_SEED=$C --MEMORY_SHARE=0.95 --DATASET=$DATASET --ITERS=$ITERS --BATCH_SIZE=$BS --LEARNING_RATE=$LR --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --WIDENESS=$WIDENESS --DO_BATCHNORM=$BN --TRAIN_DATASET_SIZE=$TRAIN --AUGMENTATION=$AUGMENTATION"

    echo BASELINE
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS > couts/$NAME.BASELINE_${C}.cout 2> couts/$NAME.BASELINE_${C}.cerr

    echo DATAGRAD
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DG > couts/$NAME.DG_${DG}_${C}.cout 2> couts/$NAME.DG_${DG}_${C}.cerr

    echo SPECTREG
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA > couts/$NAME.LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.LAMBDA_${LAMBDA}_${C}.cerr

done

echo "DONE"
