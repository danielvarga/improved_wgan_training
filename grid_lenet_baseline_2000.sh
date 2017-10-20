NAME=lenet_baseline_2000
mkdir -p couts/

for WD in 0 0.00001 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1
do
    for DROPOUT in `seq 0.1 0.1 1.0`
    do
		CUDA_VISIBLE_DEVICES=1 python classifier.py --DROPOUT_KEEP_PROB=$DROPOUT --WEIGHT_DECAY=$WD --DISC_TYPE=lenet --DATASET=mnist --TRAIN_DATASET_SIZE=2000 --LEARNING_RATE=0.001 > couts/$NAME.WD_${WD}_DROPOUT_${DROPOUT}.cout 2> couts/$NAME.TRAIN_WD_${WD}_DROPOUT_${DROPOUT}.cerr
	done
done
