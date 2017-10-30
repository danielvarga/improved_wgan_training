NAME=lenet_final
mkdir -p couts/

for C in 1 2 3
do
	for TRAINSIZE in 100 500 1000 2000 3000 4000 5000 10000
	do
		# 1a baseline with dropout
		CUDA_VISIBLE_DEVICES=1 python classifier.py --TRAIN_DATASET_SIZE=$TRAINSIZE --WEIGHT_DECAY=0.0005 > couts/$NAME.1a_TRAIN_${TRAINSIZE}_${C}.cout 2> couts/$NAME.1a_TRAIN_${TRAINSIZE}_${C}.cerr

		# 1b baseline with bn
		CUDA_VISIBLE_DEVICES=1 python classifier.py --TRAIN_DATASET_SIZE=$TRAINSIZE --WEIGHT_DECAY=0.0005 --DO_BATCHNORM=True > couts/$NAME.1b_TRAIN_${TRAINSIZE}_${C}.cout 2> couts/$NAME.1b_TRAIN_${TRAINSIZE}_${C}.cerr

		# 2 datagrad
		CUDA_VISIBLE_DEVICES=1 python classifier.py --TRAIN_DATASET_SIZE=$TRAINSIZE --WEIGHT_DECAY=0.0005 --DATAGRAD=50 > couts/$NAME.2_TRAIN_${TRAINSIZE}_${C}.cout 2> couts/$NAME.2_TRAIN_${TRAINSIZE}_${C}.cerr

		for LAMBDA in 0.00001 0.0001 0.001 0.01 0.1 1 10
		do
			# 3a GP with L2 gradient loss
			CUDA_VISIBLE_DEVICES=1 python classifier.py --TRAIN_DATASET_SIZE=$TRAINSIZE --WEIGHT_DECAY=0.0005 --DO_BATCHNORM=True --LAMBDA=$LAMBDA > couts/$NAME.3a_TRAIN_${TRAINSIZE}_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.3a_TRAIN_${TRAINSIZE}_LAMBDA_${LAMBDA}_${C}.cerr

			# 4 GP with softmax
			CUDA_VISIBLE_DEVICES=1 python classifier.py --TRAIN_DATASET_S IZE=$TRAINSIZE --WEIGHT_DECAY=0.0005 --DO_BATCHNORM=True --LAMBDA=$LAMBDA --COMBINE_OUTPUTS_MODE=softmax > couts/$NAME.4_TRAIN_${TRAINSIZE}_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.4_TRAIN_${TRAINSIZE}_LAMBDA_${LAMBDA}_${C}.cerr


			for LIPS in 0.3 0.5 0.7 1 2 3 4 5
			do
				# 3b GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET
				CUDA_VISIBLE_DEVICES=1 python classifier.py --TRAIN_DATASET_SIZE=$TRAINSIZE --WEIGHT_DECAY=0.0005 --DO_BATCHNORM=True --LAMBDA=$LAMBDA --LIPSCHITZ_TARGET==$LIPS > couts/$NAME.3b_TRAIN_${TRAINSIZE}_LAMBDA_${LAMBDA}_LIPS_${LIPS}_${C}.cout 2> couts/$NAME.3B_TRAIN_${TRAINSIZE}_LAMBDA_${LAMBDA}_LIPS_${LIPS}_${C}.cerr

    
			done
		done
	done
done

# great great comparison
