NAME=lambda_mnist
VAR=LAMBDA

mkdir -p couts/
cp classifier.py couts/$NAME.classifier.py
for VAL in `seq 0.0001 0.0001 0.0009` `seq 0.001 0.001 0.009` `seq 0.01 0.01 0.09` `seq 0.1 0.1 0.9` `seq 1 1 20`
do
	CUDA_VISIBLE_DEVICES=0 python classifier.py --ITERS=10000 --DO_BATCHNORM=False --TRAIN_DATASET_SIZE=500 --DATASET=mnist --DISC_TYPE=conv --LIPSCHITZ_TARGET=0.3 --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
done
