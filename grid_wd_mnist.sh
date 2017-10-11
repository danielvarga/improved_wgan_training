NAME=wd_mnist
VAR=WEIGHT_DECAY

mkdir -p couts/
cp classifier.py couts/$NAME.classifier.py
for VAL in `seq 0.0001 0.0002 0.0009` `seq 0.001 0.002 0.009` `seq 0.01 0.02 0.09` `seq 0.1 0.2 0.9`
do
	CUDA_VISIBLE_DEVICES=1 python classifier.py --DO_BATCHNORM=True --TRAIN_DATASET_SIZE=500 --DATASET=mnist --DISC_TYPE=conv --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
done
