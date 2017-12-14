NAME=cifar10_aug
mkdir -p couts/
echo $NAME

D=0
DATASET=cifar10
ITERS=50000
BS=128
LR=0.1
LRD=piecewise
WD=0.003
NET=cifarResnet
WIDENESS=3
BN=True
TRAIN=2000
AUGMENTATION=True

COMMON_ARGS="--MEMORY_SHARE=0.95 --DATASET=$DATASET --ITERS=$ITERS --BATCH_SIZE=$BS --LEARNING_RATE=$LR --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --WIDENESS=$WIDENESS --DO_BATCHNORM=$BN --TRAIN_DATASET_SIZE=$TRAIN --AUGMENTATION=$AUGMENTATION"

for C in `seq 1 1 10`
do
	echo ITER $C
	for DG in 0.001 0.003 0.01 0.03
	do
		echo DATAGRAD $DG
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --RANDOM_SEED=$C --DATAGRAD=$DG > couts/$NAME.DG_${DG}_${C}.cout 2> couts/$NAME.DG_${DG}_${C}.cerr
	done
	
	for EP in 0.001 0.003 0.03 0.1
	do
		echo ENTREG $EP
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --RANDOM_SEED=$C --ENTROPY_PENALTY=$EP > couts/$NAME.EP_${EP}_${C}.cout 2> couts/$NAME.EP_${EP}_${C}.cerr
	done

	for LAMBDA in 0.0001 0.0003 0.001 0.003
	do
		echo SPECTREG $EP
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --RANDOM_SEED=$C --LAMBDA=$LAMBDA > couts/$NAME.LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.LAMBDA_${LAMBDA}_${C}.cerr
	done

done

echo "DONE"