NAME=cifar10_datagrad
mkdir -p couts/
echo $NAME

D=1
DATASET=cifar10
ITERS=200000
BS=128
LR=0.1
LRD=piecewise
WD=0.0001
NET=cifarResnet
BN=True
TRAIN=50000
AUGMENTATION=1
WIDENESS=3

for C in `seq 1 1 10`
do
    echo ITER $C
    COMMON_ARGS="--MEMORY_SHARE=0.95 --DATASET=$DATASET --ITERS=$ITERS --BATCH_SIZE=$BS --LEARNING_RATE=$LR --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --WIDENESS=$WIDENESS --DO_BATCHNORM=$BN --TRAIN_DATASET_SIZE=$TRAIN --AUGMENTATION=$AUGMENTATION"

    for DATAGRAD in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1
    do
	echo DATAGRAD $DATAGRAD
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --RANDOM_SEED=$C --DATAGRAD=$DATAGRAD > couts/$NAME.DATAGRAD_${DATAGRAD}_${C}.cout 2> couts/$NAME.DATAGRAD_${DATAGRAD}_${C}.cerr
    done

done

echo "DONE"