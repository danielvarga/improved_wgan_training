NAME=lenet_piecewise
mkdir -p couts/

for C in 1 2 3 4 5 6 7 8 9 10
do
	for TRAIN_DATASET_SIZE in 500 1000 2000 3000 4000 5000
	do
		CUDA_VISIBLE_DEVICES=0 python classifier.py --TRAIN_DATASET_SIZE=$TRAIN_DATASET_SIZE --LEARNING_RATE_DECAY=piecewise --LEARNING_RATE=0.001 --DISC_TYPE=lenet --DATASET=mnist --WEIGHT_DECAY=0.0005 > couts/$NAME.TRAIN_${TRAIN_DATASET_SIZE}_${C}_unreg.cout 2> couts/$NAME.TRAIN_${TRAIN_DATASET_SIZE}_${C}_unreg.cerr

		CUDA_VISIBLE_DEVICES=0 python classifier.py --LAMBDA=0.0001 --TRAIN_DATASET_SIZE=$TRAIN_DATASET_SIZE --LEARNING_RATE_DECAY=piecewise --LEARNING_RATE=0.001 --DISC_TYPE=lenet --DATASET=mnist --WEIGHT_DECAY=0.0005 > couts/$NAME.TRAIN_${TRAIN_DATASET_SIZE}_${C}_gp.cout 2> couts/$NAME.TRAIN_${TRAIN_DATASET_SIZE}_${C}_gp.cerr

		CUDA_VISIBLE_DEVICES=0 python classifier.py --DATAGRAD=50 --TRAIN_DATASET_SIZE=$TRAIN_DATASET_SIZE --LEARNING_RATE_DECAY=piecewise --LEARNING_RATE=0.001 --DISC_TYPE=lenet --DATASET=mnist --WEIGHT_DECAY=0.0005 > couts/$NAME.TRAIN_${TRAIN_DATASET_SIZE}_${C}_datagrad.cout 2> couts/$NAME.TRAIN_${TRAIN_DATASET_SIZE}_${C}_datagrad.cerr

		CUDA_VISIBLE_DEVICES=0 python classifier.py --COMBINE_OUTPUTS_MODE=softmax --GP_VERSION=3 --LAMBDA=0.001 --TRAIN_DATASET_SIZE=$TRAIN_DATASET_SIZE --LEARNING_RATE_DECAY=piecewise --LEARNING_RATE=0.001 --DISC_TYPE=lenet --DATASET=mnist --WEIGHT_DECAY=0.0005 > couts/$NAME.TRAIN_${TRAIN_DATASET_SIZE}_${C}_softmax.cout 2> couts/$NAME.TRAIN_${TRAIN_DATASET_SIZE}_${C}_softmax.cerr
    
	done
done

# unreg vs gp vs datagrad vs softmax
