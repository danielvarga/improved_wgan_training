# Find optimal datagrad values for a large range of trainsizes

NAME=mnist_datagrad
mkdir -p couts/
echo $NAME

D=1
WD=0.0005
LRD=piecewise

for C in `seq 1 1 10`
do
    echo ITER $C
	for NET in lenet lenettuned
	do
		echo NET $NET
		for TRAIN in 500 1000 2000 3000 4000 5000 10000 15000 20000
		do
			COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN"

			for DATAGRAD in 0 1 2 5 10 20 50 100 200 500
			do
				CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DATAGRAD > couts/$NAME.do_DG_${DATAGRAD}_NET_${NET}_${C}.cout 2> couts/$NAME.do_DG_${DATAGRAD}_NET_${NET}_${C}.cerr
			done
		done
	done
done
