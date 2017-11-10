# compare batchnorm, dropout and nothing vs Gradient Regularization
# for unreg: DATAGRAD = 50, LAMBDA=0.01
# for dropout: DATAGRAD = 50, LAMBDA=0.01
# for batchnorm: DATAGRAD = 0.001, LAMBDA=0.001

NAME=mnist_1
mkdir -p couts/
echo $NAME

D=1
WD=0.0005
TRAIN=2000
LRD=piecewise
NET=lenettuned

# additional loop for finding DATAGRAD for batchnorm

for C in `seq 1 1 10`
do
    echo ITER $C
	COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN"

	for DATAGRAD in 0.001 0.003 0.01 0.03 0.1 0.3 1
	do
		# bn
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DATAGRAD --DO_BATCHNORM=True > couts/$NAME.bn_DG_${DATAGRAD}_${C}.cout 2> couts/$NAME.bn_DG_${DATAGRAD}_${C}.cerr
	done	
done


for C in `seq 1 1 10`
do
    echo ITER $C
	COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN"

	for DATAGRAD in 0 5 10 20 50
	do
		# baseline no dropout, no bn
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DATAGRAD --DO_BATCHNORM=False --DROPOUT_KEEP_PROB=1 > couts/$NAME.unreg_DG_${DATAGRAD}_${C}.cout 2> couts/$NAME.unreg_DG_${DATAGRAD}_${C}.cerr

		# bn
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DATAGRAD --DO_BATCHNORM=True > couts/$NAME.bn_DG_${DATAGRAD}_${C}.cout 2> couts/$NAME.bn_DG_${DATAGRAD}_${C}.cerr

		# dropout
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DATAGRAD --DO_BATCHNORM=False > couts/$NAME.do_DG_${DATAGRAD}_${C}.cout 2> couts/$NAME.do_DG_${DATAGRAD}_${C}.cerr
	done

	for LAMBDA in 0.001 0.003 0.01 0.03
	do
		# baseline no dropout, no bn
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA --DO_BATCHNORM=False --DROPOUT_KEEP_PROB=1  > couts/$NAME.unreg_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.unreg_LAMBDA_${LAMBDA}_${C}.cerr

		# bn
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA --DO_BATCHNORM=True > couts/$NAME.bn_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.bn_LAMBDA_${LAMBDA}_${C}.cerr

		# dropout
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA --DO_BATCHNORM=False > couts/$NAME.do_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.do_LAMBDA_${LAMBDA}_${C}.cerr
	done
	
done

