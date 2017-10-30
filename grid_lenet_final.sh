NAME=lenet_final
mkdir -p couts/

D=0
WD=0.0005

for C in 1 2
do
	for TRAIN in 100 500 1000 2000 3000 4000 5000 10000
	do
		echo TRAIN $TRAIN
		for BN in True False
		do
			echo -BN $BN
			# 1 baseline unreg
			CUDA_VISIBLE_DEVICES=$D python classifier.py --WEIGHT_DECAY=$WD --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN > couts/$NAME.1_TRAIN_${TRAIN}_BN_${BN}_${C}.cout 2> couts/$NAME.1_TRAIN_${TRAIN}_BN_${BN}_${C}.cerr

			# 2 datagrad
			for DG in 0.01 0.1 1 10 50 10
			do
				echo --DG $DG
				CUDA_VISIBLE_DEVICES=$D python classifier.py --WEIGHT_DECAY=$WD --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN --DATAGRAD=$DG > couts/$NAME.2_TRAIN_${TRAIN}_BN_${BN}_DG_${DG}_${C}.cout 2> couts/$NAME.2_TRAIN_${TRAIN}_BN_${BN}_DG_${DG}_${C}.cerr
			done

			# GP
			for LAMBDA in 0.0001 0.001 0.01 0.1 1
			do
				echo --LAMBDA $LAMBDA
				# 3a GP with L2 gradient loss
				CUDA_VISIBLE_DEVICES=$D python classifier.py --WEIGHT_DECAY=$WD --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN --LAMBDA=$LAMBDA --GP_VERSION=3 > couts/$NAME.3a_TRAIN_${TRAIN}_BN_${BN}_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.3a_TRAIN_${TRAIN}_BN_${BN}_LAMBDA_${LAMBDA}_${C}.cerr

				# 4 GP with softmax
				CUDA_VISIBLE_DEVICES=$D python classifier.py --WEIGHT_DECAY=$WD --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN --LAMBDA=$LAMBDA --GP_VERSION=3 --COMBINE_OUTPUTS_MODE=softmax > couts/$NAME.4_TRAIN_${TRAIN}_BN_${BN}_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.4_TRAIN_${TRAIN}_BN_${BN}_LAMBDA_${LAMBDA}_${C}.cerr


				# 3b GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET
				for LIPS in 0.3 0.5 0.7 1 2 3 4 5
				do
					echo --LIPS $LIPS
					CUDA_VISIBLE_DEVICES=$D python classifier.py --WEIGHT_DECAY=$WD --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=$BN --LAMBDA=$LAMBDA --GP_VERSION=4 --LIPSCHITZ_TARGET=$LIPS > couts/$NAME.3b_TRAIN_${TRAIN}_BN_${BN}_LAMBDA_${LAMBDA}_LIPS_${LIPS}_${C}.cout 2> couts/$NAME.3b_TRAIN_${TRAIN}_BN_${BN}_LAMBDA_${LAMBDA}_LIPS_${LIPS}_${C}.cerr
				done
			done
		done
	done
done

# great great comparison
