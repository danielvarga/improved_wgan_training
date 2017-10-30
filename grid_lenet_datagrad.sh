NAME=lenet_datagrad
VAR=DATAGRAD

mkdir -p couts/
for C in 1 2 3 4 5 6 7 8 9 10
do
	for VAL in 0.01 0.1 1 10 50 100 150 200
	# for VAL in `seq 10 10 100`
	do
		CUDA_VISIBLE_DEVICES=1 python classifier.py --DO_BATCHNORM=False --WEIGHT_DECAY=0.0005 --DATASET=mnist --LEARNING_RATE=0.001 --MEMORY_SHARE=0.4 --DISC_TYPE=lenet --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
		CUDA_VISIBLE_DEVICES=1 python classifier.py --DO_BATCHNORM=True --WEIGHT_DECAY=0.0005 --DATASET=mnist --LEARNING_RATE=0.001 --MEMORY_SHARE=0.4 --DISC_TYPE=lenet --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
	done
done

# conclusion: DATAGRAD in (10, 50) is the best range
# accuracy is around 97.5%
# this is more than half percent increase compared to the baseline

# datagrad works much better with dropout than with batchnorm
