NAME=wd_50000
VAR=WEIGHT_DECAY

mkdir -p couts/
cp classifier.py couts/$NAME.classifier.py
for VAL in 0 0.0001 0.001
do
	CUDA_VISIBLE_DEVICES=1 python classifier.py --LEARNING_RATE_DECAY=True --LEARNING_RATE=0.05 --TRAIN_DATASET_SIZE=50000 --ITERS=20000 --$VAR=$VAL > couts/$NAME.1_${VAR}_${VAL}.cout 2> couts/$NAME.1_${VAR}_${VAL}.cerr
done
