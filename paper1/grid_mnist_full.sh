# compare dg and ep on full mnist

NAME=mnist_full
mkdir -p couts/

D=0
WD=0.0005
TRAIN=50000
LRD=piecewise
NET=lenet
BN=False
EP=0.01
DG=10

for C in `seq 1 1 10`
do
    echo ITER $C
	COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN"

    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS > couts/$NAME.do_${C}.cout 2> couts/$NAME.do_${C}.cerr
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --ENTROPY_PENALTY=$EP > couts/$NAME.do_EP_${EP}_${C}.cout 2> couts/$NAME.do_EP_${EP}_${C}.cerr
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --ENTROPY_PENALTY=$EP --DATAGRAD=$DG > couts/$NAME.do_EP_${EP}_DG_${DG}_${C}.cout 2> couts/$NAME.do_EP_${EP}_DG_${DG}_${C}.cerr
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DG > couts/$NAME.do_DG_${DG}_${C}.cout 2> couts/$NAME.do_DG_${DG}_${C}.cerr
done
