NAME=trainsize
VAR=TRAIN_DATASET_SIZE

mkdir -p couts/
for VAL in `seq 100 100 900` `seq 1000 1000 10000`
do
	CUDA_VISIBLE_DEVICES=0 python classifier.py --LEARNING_RATE=0.1 --ITERS=10000 --$VAR=$VAL > couts/$NAME.1_${VAR}_${VAL}.cout 2> couts/$NAME.1_${VAR}_${VAL}.cerr
	CUDA_VISIBLE_DEVICES=0 python classifier.py --LEARNING_RATE_DECAY=True --LEARNING_RATE=0.1 --ITERS=10000 --$VAR=$VAL > couts/$NAME.2_${VAR}_${VAL}.cout 2> couts/$NAME.2_${VAR}_${VAL}.cerr
done
