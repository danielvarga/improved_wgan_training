NAME=lipschitz_mnist_gs
VAR=LIPSCHITZ_TARGET

mkdir -p couts/
cp classifier.py couts/$NAME.classifier.py
for VAL in `seq 0.1 0.1 0.9` `seq 1 1 20`
do
	CUDA_VISIBLE_DEVICES=1 python classifier.py --GRADIENT_SHRINKING=True --ITERS=10000 --DO_BATCHNORM=False --TRAIN_DATASET_SIZE=500 --DATASET=mnist --DISC_TYPE=conv --LAMBDA=0 --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
done
