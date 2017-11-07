NAME=lenet_grandfinal
mkdir -p couts/

D=1
WD=0.0005
BN=False
LAMBDA=0.01
LIPS=0.7
LRD=piecewise

for C in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    echo ITER $C
    for TRAIN in 500 1000 2000 3000 4000 5000 10000
    do
	# 1 baseline unreg
	CUDA_VISIBLE_DEVICES=$D python classifier.py --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN > couts/$NAME.1_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.1_TRAIN_${TRAIN}_${C}.cerr
	# 2 datagrad
	CUDA_VISIBLE_DEVICES=$D python classifier.py --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN --DATAGRAD=10 > couts/$NAME.2_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.2_TRAIN_${TRAIN}_${C}.cerr

	# 3a GP with L2 gradient loss
	CUDA_VISIBLE_DEVICES=$D python classifier.py --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN --LAMBDA=$LAMBDA --GP_VERSION=3 > couts/$NAME.3a_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.3a_TRAIN_${TRAIN}_${C}.cerr

	# 4 GP with softmax
	CUDA_VISIBLE_DEVICES=$D python classifier.py --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN --LAMBDA=0.1 --GP_VERSION=3 --COMBINE_OUTPUTS_MODE=softmax > couts/$NAME.4_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.4_TRAIN_${TRAIN}_${C}.cerr

	
	# 3b GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET
	CUDA_VISIBLE_DEVICES=$D python classifier.py --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN --LAMBDA=$LAMBDA --GP_VERSION=4 --LIPSCHITZ_TARGET=$LIPS > couts/$NAME.3b_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.3b_TRAIN_${TRAIN}_${C}.cerr
    done
done

# great great comparison
