# tuning CIFAR-10 3-wide N=3 resnet to find optimal lambas

NAME=cifar10_tune
mkdir -p couts/
echo $NAME

D=1
WD=0.003
TRAIN=50000
LRD=piecewise
NET=cifarResnet
WIDENESS=3
LEARNING_RATE=0.1
ITERS=50000
BS=128
BN=True

for C in `seq 1 1 2`
do
    echo ITER $C
	COMMON_ARGS="--RANDOM_SEED=$C --MEMORY_SHARE=0.95 --DATASET=cifar10 --ITERS=$ITERS --BATCH_SIZE=$BS --LEARNING_RATE=$LEARNING_RATE --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --WIDENESS=$WIDENESS --DO_BATCHNORM=$BN --TRAIN_DATASET_SIZE=$TRAIN"

	echo SpectReg
	for W in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1
	do
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$W > couts/$NAME.SpectReg_W_${W}_${C}.cout 2> couts/$NAME.SpectReg_W_${W}_${C}.cerr
	done

	echo EntReg
	for W in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1
	do
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --ENTROPY_PENALTY=$W > couts/$NAME.EntReg_W_${W}_${C}.cout 2> couts/$NAME.EntReg_W_${W}_${C}.cerr
	done

	echo JacReg
	for W in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1
	do
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$W --COMBINE_OUTPUTS_FOR_SLOPES=False --COMBINE_OUTPUTS_MODE=softmax > couts/$NAME.JacReg_W_${W}_${C}.cout 2> couts/$NAME.JacReg_W_${W}_${C}.cerr
	done

	echo FrobReg
	for W in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1
	do
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$W --COMBINE_OUTPUTS_MODE=False > couts/$NAME.FrobReg_W_${W}_${C}.cout 2> couts/$NAME.FrobReg_W_${W}_${C}.cerr
	done

	echo Onehot
	for W in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1
	do
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$W --COMBINE_OUTPUTS_MODE=onehot > couts/$NAME.Onehot_W_${W}_${C}.cout 2> couts/$NAME.Onehot_W_${W}_${C}.cerr
	done

	echo RandomOnehot
	for W in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1
	do
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$W --COMBINE_OUTPUTS_MODE=random_onehot > couts/$NAME.RandomOnehot_W_${W}_${C}.cout 2> couts/$NAME.RandomOnehot_W_${W}_${C}.cerr
	done

	echo DataGrad
	for W in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1
	do
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$W > couts/$NAME.DataGrad_W_${W}_${C}.cout 2> couts/$NAME.DataGrad_W_${W}_${C}.cerr
	done

done

echo "DONE"
