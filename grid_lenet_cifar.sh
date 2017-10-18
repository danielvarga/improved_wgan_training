NAME=lenet_cifar
VAR=TRAIN_DATASET_SIZE

mkdir -p couts/
for VAL in `seq 100 100 900` `seq 1000 1000 10000`
do
	CUDA_VISIBLE_DEVICES=0 python classifier.py --LEARNING_RATE=0.01 --DISC_TYPE=conv2 --LEARNING_RATE_DECAY=piecewise --ITERS=20000 --$VAR=$VAL > couts/$NAME.1_${VAR}_${VAL}.cout 2> couts/$NAME.1_${VAR}_${VAL}.cerr
	CUDA_VISIBLE_DEVICES=0 python classifier.py --LAMBDA=0.0001 --LIPSCHITZ_TARGET=0.9 --LEARNING_RATE=0.01 --DISC_TYPE=conv2 --LEARNING_RATE_DECAY=piecewise --ITERS=20000 --$VAR=$VAL > couts/$NAME.1_${VAR}_${VAL}.cout 2> couts/$NAME.1_${VAR}_${VAL}.cerr
	CUDA_VISIBLE_DEVICES=0 python classifier.py --GRADIENT_SHRINKING=True --LIPSCHITZ_TARGET=2 --LEARNING_RATE=0.01 --DISC_TYPE=conv2 --LEARNING_RATE_DECAY=piecewise --ITERS=20000 --$VAR=$VAL > couts/$NAME.1_${VAR}_${VAL}.cout 2> couts/$NAME.1_${VAR}_${VAL}.cerr
done
