# find best weight for entropy penalty

NAME=mnist_ent
mkdir -p couts/
echo $NAME

D=0
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
	
			for EP in 0.001 0.003 0.01 0.03 0.1 0.3 
			do
				CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --ENTROPY_PENALTY=$EP > couts/$NAME.do_EP_${EP}_NET_${NET}_${C}.cout 2> couts/$NAME.do_EP_${EP}_NET_${NET}_${C}.cerr
			done
		done
	done
done

