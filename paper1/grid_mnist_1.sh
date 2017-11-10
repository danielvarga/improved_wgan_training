# compare batchnorm, dropout and nothing vs Gradient Regularization

NAME=mnist_1
mkdir -p couts/

D=1
WD=0.0005
TRAIN=2000
LRD=piecewise
NET=lenet

for C in `seq 1 1 10`
do
    echo ITER $C
	COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN"

	for DATAGRAD in 0 5 10 20 50
	do
		# baseline no dropout, no bn
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DO_BATCHNORM=False --DROPOUT_KEEP_PROB=1  > couts/$NAME.unreg_DG_${DATAGRAD}_${C}.cout 2> couts/$NAME.unreg_DG_${DATAGRAD}_${C}.cerr

		# bn
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DO_BATCHNORM=True > couts/$NAME.bn_DG_${DATAGRAD}_${C}.cout 2> couts/$NAME.bn_DG_${DATAGRAD}_${C}.cerr

		# dropout
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DO_BATCHNORM=False > couts/$NAME.do_DG_${DATAGRAD}_${C}.cout 2> couts/$NAME.do_DG_${DATAGRAD}_${C}.cerr
	done

	for LAMBDA in 0.001 0.003 0.01 0.03
	do
		# baseline no dropout, no bn
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DO_BATCHNORM=False --DROPOUT_KEEP_PROB=1  > couts/$NAME.unreg_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.unreg_LAMBDA_${LAMBDA}_${C}.cerr

		# bn
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DO_BATCHNORM=True > couts/$NAME.bn_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.bn_LAMBDA_${LAMBDA}_${C}.cerr

		# dropout
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DO_BATCHNORM=False > couts/$NAME.do_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.do_LAMBDA_${LAMBDA}_${C}.cerr
	done
	
done

