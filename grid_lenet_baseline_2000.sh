NAME=lenet_baseline_2000
mkdir -p couts/

for DROPOUT in 0.25 0.5 0.75
do
	for WD in 0 0.00001 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1    
    do
		CUDA_VISIBLE_DEVICES=1 python classifier.py --DROPOUT_KEEP_PROB=$DROPOUT --WEIGHT_DECAY=$WD --DISC_TYPE=lenet --DATASET=mnist --TRAIN_DATASET_SIZE=2000 --LEARNING_RATE=0.001 > couts/$NAME.WD_${WD}_DROPOUT_${DROPOUT}.cout 2> couts/$NAME.TRAIN_WD_${WD}_DROPOUT_${DROPOUT}.cerr
	done
done

# conclusion: wd=0.0005 dropout=0.5 is a good setting (just like in full baseline)
# accuracy is close to 97% (slightly under)
