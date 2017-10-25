NAME=lenet_gp_fashion
mkdir -p couts/

for LAMBDA in 0.0001 0.001 0.01 0.1 1
do
	for LIPSCHITZ_TARGET in 0.3 0.5 0.7 0.9 1 2 3 4 5
    do
		CUDA_VISIBLE_DEVICES=1 python classifier.py --LAMBDA=$LAMBDA --LIPSCHITZ_TARGET=$LIPSCHITZ_TARGET --DROPOUT_KEEP_PROB=0.5 --WEIGHT_DECAY=0.1 --DISC_TYPE=lenet --DATASET=fashion_mnist --TRAIN_DATASET_SIZE=2000 --LEARNING_RATE=0.001 > couts/$NAME.LAMBDA_${LAMBDA}_LIPSCHITZ_${LIPSCHITZ_TARGET}.cout 2> couts/$NAME.LAMBDA_${LAMBDA}_LIPSCHITZ_${LIPSCHITZ_TARGET}.cerr
	done
done

# running it with batchnorm instead of dropout
# accuracy is slightly under 84% (not better than the baseline)
# LAMBDA=0.1, LIPSCHITZ_TARGET=3
