NAME=trainsize_mnist_gsgp
VAR=TRAIN_DATASET_SIZE

mkdir -p couts/
cp classifier.py couts/$NAME.classifier.py
for VAL in `seq 100 100 900` `seq 1000 1000 10000`
do
	CUDA_VISIBLE_DEVICES=0 python classifier.py --GRADIENT_SHRINKING=True --LAMBDA=0.0001 --DO_BATCHNORM=False --DATASET=mnist --DISC_TYPE=conv --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
done
