# tuning CIFAR-10 3-wide N=3 resnet to find optimal lambas for trainsize=2000, iters=10000

NAME=cifar10_tune2000
mkdir -p couts/
echo $NAME

D=0
TRAIN=2000
LRD=piecewise
NET=cifarResnet
WIDENESS=3
LEARNING_RATE=0.1
ITERS=10000
BS=128
BN=True

for C in `seq 1 1 1`
do
    echo ITER $C
	for AUG in True False
	do
		for WD in 0.0001
		do
			COMMON_ARGS="--RANDOM_SEED=$C --MEMORY_SHARE=0.95 --DATASET=cifar10 --ITERS=$ITERS --BATCH_SIZE=$BS --LEARNING_RATE=$LEARNING_RATE --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --WIDENESS=$WIDENESS --DO_BATCHNORM=$BN --TRAIN_DATASET_SIZE=$TRAIN --AUGMENTATION=$AUG"

			# baseline
			CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS > couts/$NAME.unreg_aug_${AUG}_WD_${WD}_${C}.cout 2> couts/$NAME.unreg_aug_${AUG}_WD_${WD}_${C}.cerr

			echo SpectReg
			for W in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1
			do
				CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$W > couts/$NAME.SpectReg_W_${W}_${AUG}_WD_${WD}_${C}.cout 2> couts/$NAME.SpectReg_W_${W}_${AUG}_WD_${WD}_${C}.cerr
			done

			echo DataGrad
			for W in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1
			do
				CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$W > couts/$NAME.DataGrad_W_${W}_${AUG}_WD_${WD}_${C}.cout 2> couts/$NAME.DataGrad_W_${W}_${AUG}_WD_${WD}_${C}.cerr
			done

			echo EntReg
			for W in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1
			do
				CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --ENTROPY_PENALTY=$W > couts/$NAME.EntReg_W_${W}_${AUG}_WD_${WD}_${C}.cout 2> couts/$NAME.EntReg_W_${W}_${AUG}_WD_${WD}_${C}.cerr
			done
		done
	done
done

echo "DONE"
