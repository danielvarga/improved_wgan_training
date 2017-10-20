NAME=lenet_softmax
VAR=LAMBDA

mkdir -p couts/
for VAL in 0.0001 0.001 0.01 0.1 1 10
do
	CUDA_VISIBLE_DEVICES=1 python classifier.py --COMBINE_OUTPUTS_MODE=softmax --GP_VERSION=3 --DROPOUT_KEEP_PROB=0.5 --WEIGHT_DECAY=0.0005 --DATASET=mnist --LEARNING_RATE=0.001 --MEMORY_SHARE=0.4 --DISC_TYPE=lenet --ITERS=10000 --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
done

# conclusion: this fails, but replacing dropout with batchnorm makes it competitive with datagrad
# accuracy (with batchnorm) is around 97.3
# BEST LAMBDA seems to be in the 0.001 - 0.1 range
