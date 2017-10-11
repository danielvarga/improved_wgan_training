NAME=lr
VAR=LEARNING_RATE

mkdir -p couts/
for VAL in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1
do
	CUDA_VISIBLE_DEVICES=1 python classifier.py --DO_BATCHNORM=True --TRAIN_DATASET_SIZE=500 --DATASET=mnist --DISC_TYPE=conv --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
	CUDA_VISIBLE_DEVICES=1 python classifier.py --LAMBDA=0.0001 --DO_BATCHNORM=False --TRAIN_DATASET_SIZE=500 --DATASET=mnist --DISC_TYPE=conv --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
	CUDA_VISIBLE_DEVICES=1 python classifier.py --GRADIENT_SHRINKING=True --DO_BATCHNORM=False --TRAIN_DATASET_SIZE=500 --DATASET=mnist --DISC_TYPE=conv --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
done

for VAL in `seq 0.01 0.01 0.1`
do
	CUDA_VISIBLE_DEVICES=1 python classifier.py --DO_BATCHNORM=True --LEARNING_RATE_DECAY=True --TRAIN_DATASET_SIZE=500 --DATASET=mnist --DISC_TYPE=conv --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
	CUDA_VISIBLE_DEVICES=1 python classifier.py --LAMBDA=0.0001 --DO_BATCHNORM=False --LEARNING_RATE_DECAY=True --TRAIN_DATASET_SIZE=500 --DATASET=mnist --DISC_TYPE=conv --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
	CUDA_VISIBLE_DEVICES=1 python classifier.py --GRADIENT_SHRINKING=True --DO_BATCHNORM=False --LEARNING_RATE_DECAY=True --TRAIN_DATASET_SIZE=500 --DATASET=mnist --DISC_TYPE=conv --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
done
