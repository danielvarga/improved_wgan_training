# compare dg and ep for various trainsizes
# dg+ent > dg > ent > unreg

NAME=mnist_2
mkdir -p couts/

D=1
WD=0.0005
LRD=piecewise
NET=lenet
BN=False
EP=0.01
DG=50
DG2=10
TRAIN=2000

for C in `seq 1 1 5`
do
    echo ITER $C
	COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN"
	
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS > couts/$NAME.do_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.do_TRAIN_${TRAIN}_${C}.cerr
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --ENTROPY_PENALTY=$EP > couts/$NAME.do_EP_${EP}_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.do_EP_${EP}_TRAIN_${TRAIN}_${C}.cerr
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --ENTROPY_PENALTY=$EP --DATAGRAD=$DG2 > couts/$NAME.do_EP_${EP}_DG_${DG2}_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.do_EP_${EP}_DG_${DG2}_TRAIN_${TRAIN}_${C}.cerr
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DG > couts/$NAME.do_DG_${DG}_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.do_DG_${DG}_TRAIN_${TRAIN}_${C}.cerr
done
