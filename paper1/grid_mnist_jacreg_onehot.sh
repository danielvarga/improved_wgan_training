# Find optimal lambda values for jacreg and onehot

NAME=mnist_jacreg_onehot
mkdir -p couts/
echo $NAME

D=1
WD=0.0005
LRD=piecewise
TRAIN=2000

for C in `seq 1 1 10`
do
    echo ITER $C
    for NET in lenet lenettuned
    do
	echo NET $NET
	for LAMBDA in 0.0001 0.001 0.01 0.1
	do
	    COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN --LAMBDA=$LAMBDA --MEMORY_SHARE=0.95"
	
	    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --COMBINE_OUTPUTS_FOR_SLOPES=False > couts/$NAME.jacreg_LB_${LAMBDA}_NET_${NET}_${C}.cout 2> couts/$NAME.jacreg_LB_${LAMBDA}_NET_${NET}_${C}.cerr
	    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --COMBINE_OUTPUTS_MODE=onehot > couts/$NAME.onehot_LB_${LAMBDA}_NET_${NET}_${C}.cout 2> couts/$NAME.onehot_LB_${LAMBDA}_NET_${NET}_${C}.cerr
	done
    done
done

