NAME=lenet_datagrad
VAR=DATAGRAD

mkdir -p couts/
for I in 1 2 3
do
	for VAL in 0.00001 0.0001 0.001 0.01 0.1 1 10 100
#	for VAL in 150 200 500 1000 2000
	do
		CUDA_VISIBLE_DEVICES=0 python classifier.py --DATASET=mnist --LEARNING_RATE=0.001 --MEMORY_SHARE=0.4 --DISC_TYPE=lenet --LEARNING_RATE_DECAY=piecewise --ITERS=10000 --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
	done
done
