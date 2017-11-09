# find best weight for entropy penalty

NAME=mnist_ent
mkdir -p couts/

D=0
WD=0.0005
TRAIN=2000
LRD=piecewise
NET=lenet
BN=False

for C in `seq 1 1 10`
do
    echo ITER $C
	COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN"
	
	for EP in 0.001 0.003 0.01 0.03 0.1 0.3 
	do
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --ENTROPY_PENALTY=$EP > couts/$NAME.do_EP_${EP}_${C}.cout 2> couts/$NAME.do_EP_${EP}_${C}.cerr
	done
done

