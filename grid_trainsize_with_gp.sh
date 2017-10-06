NAME=trainsize
VAR=TRAIN_DATASET_SIZE

mkdir -p couts/
for VAL in `seq 100 100 900` `seq 1000 1000 10000`
do
	CUDA_VISIBLE_DEVICES=0 python classifier.py --LAMBDA=0.0001 --LIPSCHITZ_TARGET=0.3 --ITERS=20000 --DATASET=cifar10 --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
done
