NAME=mnist_wl
mkdir -p couts/

D=0
#WD=0.0005
LRD=piecewise
NET=lenet
BN=False
EP=0.01
#DG=50
#DG2=10
TRAIN=2000
LAMBDA=0.03

for C in `seq 1 1 10`
do
    echo ITER $C
    COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN"

    for WD in 0.0 0.0001 0.0005 0.001 0.01
    do
      for DG in 0.0 0.0001 0.001 0.01 0.1 1.0 10.0 100.0 1000.0
      do
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --WEIGHT_DECAY=$WD --DATAGRAD=$DG > couts/$NAME.do_DG_${DG}_WD_{$WD}_TRAIN_${TRAIN}_${C}.cout 2> couts/$NAME.do_DG_${DG}_WD_{$WD}_TRAIN_${TRAIN}_${C}.cerr
      done
    done
done
