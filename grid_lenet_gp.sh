NAME=lenet_gp
mkdir -p couts/

for LAMBDA in 0.0001 0.001 0.01 0.1 1
do
	for LIPSCHITZ_TARGET in 0.3 0.5 0.7 0.9 1 2 3 4 5
    do
		CUDA_VISIBLE_DEVICES=1 python classifier.py --LAMBDA=$LAMBDA --LIPSCHITZ_TARGET=$LIPSCHITZ_TARGET --DROPOUT_KEEP_PROB=0.5 --WEIGHT_DECAY=0.0005 --DISC_TYPE=lenet --DATASET=mnist --TRAIN_DATASET_SIZE=2000 --LEARNING_RATE=0.001 > couts/$NAME.LAMBDA_${LAMBDA}_LIPSCHITZ_${LIPSCHITZ_TARGET}.cout 2> couts/$NAME.TRAIN_LAMBDA_${LAMBDA}_LIPSCHITZ_${LIPSCHITZ_TARGET}.cerr
	done
done

# conclusion: LAMBDA should be small (0.0001 or 0.001), LIPSCHITZ_TARGET should be between 0.3 and 3
# half or the runs failed. GP with dropout might be unstable
# best results are slightly under 97% accuracy
