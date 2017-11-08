NAME=cifar_final
mkdir -p couts/

D=1
WD=0.0005
BN=True
LAMBDA=0.01
LIPS=0.7
LR=0.1
LRD=piecewise
DATASET=cifar10
DISC_TYPE=cifarResnet
MEMORY_SHARE=0.95
ITERS=50000
TRAIN=2000


for C in 1 2 3 4 5
do
    echo ITER $C
	COMMON_ARGS="--MEMORY_SHARE=$MEMORY_SHARE --RANDOM_SEED=$C --ITERS=$ITERS --DISC_TYPE=$DISC_TYPE --DATASET=$DATASET --LEARNING_RATE=$LR --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DO_BATCHNORM=$BN --TRAIN_DATASET_SIZE=$TRAIN"
	
	# 1 baseline unreg
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS > couts/$NAME.1_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.1_TRAIN_${TRAIN}_${C}.cerr

	# 2 datagrad
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=10 > couts/$NAME.2_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.2_TRAIN_${TRAIN}_${C}.cerr

	# 3a GP with L2 gradient loss
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA --GP_VERSION=3 > couts/$NAME.3a_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.3a_TRAIN_${TRAIN}_${C}.cerr

	# 4 GP with softmax
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=0.1 --GP_VERSION=3 --COMBINE_OUTPUTS_MODE=softmax > couts/$NAME.4_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.4_TRAIN_${TRAIN}_${C}.cerr

	# 3b GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA --GP_VERSION=4 --LIPSCHITZ_TARGET=$LIPS > couts/$NAME.3b_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.3b_TRAIN_${TRAIN}_${C}.cerr
done

# great great comparison
