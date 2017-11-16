# Compare onehot and random_onehot

NAME=mnist_compare2000_onehot
mkdir -p couts/
echo $NAME

D=1
WD=0.0005
LRD=piecewise
TRAIN=2000
COMBMODE=onehot

for C in `seq 1 1 10`
do
    echo ITER $C
    for NET in lenet lenettuned
    do
	echo NET $NET
	for COMBMODE in onehot random_onehot
	do
	    echo COMBMODE $COMBMODE
	    COMMON_ARGS="--MEMORY_SHARE=0.45 --RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN --COMBINE_OUTPUTS_MODE=$COMBMODE"
	    
	    for LAMBDA in 0.003 0.01 0.03 0.1 0.3
	    do
		CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA > couts/$NAME.do_LAMBDA_${LAMBDA}_NET_${NET}_combmode_${COMBMODE}_${C}.cout 2> couts/$NAME.do_LAMBDA_${LAMBDA}_NET_${NET}_combmode_${COMBMODE}_${C}.cerr
	    done
	done
    done
done
echo "DONE"
