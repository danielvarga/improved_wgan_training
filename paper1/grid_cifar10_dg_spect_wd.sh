NAME=cifar10_dg_spect_wd
mkdir -p couts/
echo $NAME

D=2
DATASET=cifar10
ITERS=50000
BS=128
LR=0.01
LRD=piecewise
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
	for WD in 0 0.001 0.003 0.01
	do
		echo "  WD" $WD
		COMMON_ARGS="--RANDOM_SEED=$C --MEMORY_SHARE=0.95 --DATASET=$DATASET --ITERS=$ITERS --BATCH_SIZE=$BS --LEARNING_RATE=$LR --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --WIDENESS=$WIDENESS --DO_BATCHNORM=$BN --TRAIN_DATASET_SIZE=$TRAIN --AUGMENTATION=$AUGMENTATION"

		echo BASELINE
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS > couts/$NAME.BASELINE_${C}.cout 2> couts/$NAME.BASELINE_${C}.cerr

		echo DATAGRAD
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DG > couts/$NAME.DG_${DG}_${C}.cout 2> couts/$NAME.DG_${DG}_${C}.cerr

		echo SPECTREG
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA > couts/$NAME.LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.LAMBDA_${LAMBDA}_${C}.cerr
	done
done

echo "DONE"