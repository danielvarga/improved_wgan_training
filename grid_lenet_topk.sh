NAME=topk
mkdir -p couts/


for C in 1 2 3 4 5 6 7 8 9 10
do
  for K in 1 2 3 4 5 6 7 8 9 10
	for TRAIN in 2000
	do
            CUDA_VISIBLE_DEVICES=0 python classifier.py --LOSS_TYPE=xent --LEARNING_RATE_DECAY=piecewise --DISC_TYPE=lenettuned --RANDOM_SEED=$C --TOPK_COMBINE_K=$K --WEIGHT_DECAY=0.0005 --TRAIN_DATASET_SIZE=$TRAIN --DO_BATCHNORM=0 --LAMBDA=0.0 --EXPLICIT_JACOBIAN=0.01 --GP_VERSION=3 --COMBINE_OUTPUTS_MODE=onehot > couts/$NAME.$C.explicit.topk$K.cout 2> couts/$NAME.$C.explicit.topk$K.cerr
	done
  done
done
