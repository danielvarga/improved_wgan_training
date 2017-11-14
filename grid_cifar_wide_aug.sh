NAME=cifar_wide_aug
mkdir -p couts/

D=2
WD=0.003
BN=True
LAMBDA=0.01
LR=0.1
LRD=piecewise
DATASET=cifar10
DISC_TYPE=cifarResnet
MEMORY_SHARE=0.95
ITERS=50000
TRAIN=2000
WIDENESS=5
VERBOSITY=2

for C in 1
do
    echo ITER $C
	COMMON_ARGS="--WIDENESS=$WIDENESS --VERBOSITY=$VERBOSITY --MEMORY_SHARE=$MEMORY_SHARE --RANDOM_SEED=$C --ITERS=$ITERS --DISC_TYPE=$DISC_TYPE --DATASET=$DATASET --LEARNING_RATE=$LR --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DO_BATCHNORM=$BN --TRAIN_DATASET_SIZE=$TRAIN"
	
	# 1 baseline unreg + aug
	#CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --AUGMENTATION=1 > couts/$NAME.1_TRAIN_${TRAIN}_unreg_aug_${C}.cout 2> couts/$NAME.1_TRAIN_${TRAIN}_unreg_aug_${C}.cerr

	# 2 datagrad + aug
	#CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --AUGMENTATION=1 --DATAGRAD=1 > couts/$NAME.2_TRAIN_${TRAIN}_dg_aug_${C}.cout 2> couts/$NAME.2_TRAIN_${TRAIN}_dg_aug_${C}.cerr

	# 2 datagrad + entropy + aug
	#CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --AUGMENTATION=1 --DATAGRAD=1 --ENTROPY_PENALTY=0.01 > couts/$NAME.2_TRAIN_${TRAIN}_dgent_aug_${C}.cout 2> couts/$NAME.2_TRAIN_${TRAIN}_dgent_aug_${C}.cerr

	# 3a GP with L2 gradient loss + aug
	#CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --AUGMENTATION=1 --LAMBDA=$LAMBDA --GP_VERSION=3 > couts/$NAME.3a_TRAIN_${TRAIN}_gpv3aug_${C}.cout 2> couts/$NAME.3a_TRAIN_${TRAIN}_gpv3_aug_${C}.cerr

	# 1 baseline unreg
	#CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS > couts/$NAME.1_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.1_TRAIN_${TRAIN}_${C}.cerr

	# 2 datagrad
	#CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=1 > couts/$NAME.2_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.2_TRAIN_${TRAIN}_${C}.cerr

	# 2 datagrad
	#CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=1 --ENTROPY_PENALTY=0.01 > couts/$NAME.2_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.2_TRAIN_${TRAIN}_${C}.cerr

	# 3a GP with L2 gradient loss
	#CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA --GP_VERSION=3 > couts/$NAME.3a_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.3a_TRAIN_${TRAIN}_${C}.cerr

        #0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0 50.0 100.0
        for DATAGRAD in 1.0 10.0
        do
		# 2 datagrad SEARCH + aug
		CUDA_VISIBLE_DEVICES=1 python classifier.py $COMMON_ARGS --AUGMENTATION=1 --DATAGRAD=$DATAGRAD > couts/$NAME.2_TRAIN_${TRAIN}_dg{$DATAGRAD}_aug_${C}_${WIDENESS}w.cout 2> couts/$NAME.2_TRAIN_${TRAIN}_dg{$DATAGRAD}_aug_${C}_${WIDENESS}w.cerr

        done

	### 4 GP with softmax
	### CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=0.1 --GP_VERSION=3 --COMBINE_OUTPUTS_MODE=softmax > couts/$NAME.4_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.4_TRAIN_${TRAIN}_${C}.cerr

	### 3b GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET
	### CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA --GP_VERSION=4 --LIPSCHITZ_TARGET=$LIPS > couts/$NAME.3b_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.3b_TRAIN_${TRAIN}_${C}.cerr
done

# great great comparison
