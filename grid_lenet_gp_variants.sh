NAME=lenet_gp_variants
mkdir -p couts/

for C in 1 2 3 4 5
do
	for GP_VERSION in 1 2 3 4
    do
		CUDA_VISIBLE_DEVICES=0 python classifier.py --LAMBDA=0.0001  --DROPOUT_KEEP_PROB=0.5 --WEIGHT_DECAY=0.0005 --DISC_TYPE=lenet --DATASET=mnist --TRAIN_DATASET_SIZE=2000 --LEARNING_RATE=0.001 --GP_VERSION=$GP_VERSION > couts/$NAME.GP_VERSION_${GP_VERSION}_${C}.cout 2> couts/$NAME.GP_VERSION_${GP_VERSION}_${C}.cerr
	done
done

# conclusion: 
# half or the runs failed. all GP variants are unstable with dropout
# best results are slightly under 97% accuracy (if it does not fail) just like the baseline
