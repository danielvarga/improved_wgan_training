# Compare DataGrad, SpectReg, EntReg, JacReg on different training sizes

NAME=mnist_4
mkdir -p couts/
echo $NAME

D=0
WD=0.0005
LRD=piecewise
NET=lenet
TRAINS=( 500 1000 2000 3000 5000 10000 15000 20000 )
DATAGRADS=( 50 50 50 20 20 5 2 2 )
EPS=( 0.01 0.01 0.01 0.01 0.1 0.1 0.01 0.03 )
LAMBDAS=( 0.03 0.03 0.03 0.03 0.003 0.01 0.01 0.001 )
JAC_LAMBDAS=( 0.3 0.03 1 1 1 1 1 0.3 )

for C in `seq 1 1 10`
do
    echo ITER $C
	echo NET $NET
	for i in `seq 1 1 8`
	do
		TRAIN=${TRAINS[i]}
		DATAGRAD=${DATAGRADS[i]}
		EP=${EPS[i]}
		LAMBDA=${LAMBDAS[i]}
		JAC_LAMBDA=${JAC_LAMBDAS[i]}

		COMMON_ARGS="--MEMORY_SHARE=0.45 --RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN"

		# SpectReg
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA > couts/$NAME.SpectReg_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.SpectReg_LAMBDA_${LAMBDA}_${C}.cerr

		# DataGrad
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DATAGRAD > couts/$NAME.DataGrad_DATAGRAD_${DATAGRAD}_${C}.cout 2> couts/$NAME.DataGrad_DATAGRAD_${DATAGRAD}_${C}.cerr

		# EntReg
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --EP=$EP > couts/$NAME.EntReg_EP_${EP}_${C}.cout 2> couts/$NAME.EntReg_EP_${EP}_${C}.cerr

		# JacReg
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA --COMBINE_OUTPUTS_FOR_SLOPES=False --COMBINE_OUTPUTS_MODE=softmax > couts/$NAME.JacReg_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.JacReg_LAMBDA_${LAMBDA}_${C}.cerr

	done
done

echo "DONE"
