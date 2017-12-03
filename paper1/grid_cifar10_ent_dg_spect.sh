NAME=cifar10_ent_dg_spect
mkdir -p couts/
echo $NAME

D=1
DATASET=cifar10
ITERS=50000
BS=128
LR=0.01
LRD=piecewise
WD=0.003
NET=cifarResnet
BN=True
TRAIN=2000
AUGMENTATION=False
WIDENESS=3
DG=1
EP=0.003
LAMBDA=0.003

for C in `seq 1 1 5`
do
    echo ITER $C
    COMMON_ARGS="--RANDOM_SEED=$C --MEMORY_SHARE=0.95 --DATASET=$DATASET --ITERS=$ITERS --BATCH_SIZE=$BS --LEARNING_RATE=$LR --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --WIDENESS=$WIDENESS --DO_BATCHNORM=$BN --TRAIN_DATASET_SIZE=$TRAIN --AUGMENTATION=$AUGMENTATION"

    echo BASELINE
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS > couts/$NAME.BASELINE_${C}.cout 2> couts/$NAME.BASELINE_${C}.cerr

    echo DATAGRAD
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DG > couts/$NAME.DG_${DG}_${C}.cout 2> couts/$NAME.DG_${DG}_${C}.cerr

    echo SPECTREG
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$DG > couts/$NAME.LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.LAMBDA_${LAMBDA}_${C}.cerr

    echo ENTREG
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --ENTROPY_PENALTY=$EP > couts/$NAME.EP_${EP}_${C}.cout 2> couts/$NAME.EP_${EP}_${C}.cerr

done

echo "DONE"
