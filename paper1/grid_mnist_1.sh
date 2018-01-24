# compare batchnorm, dropout and nothing vs Gradient Regularization
# for unreg: DATAGRAD = 50, LAMBDA=0.01
# for dropout: DATAGRAD = 50, LAMBDA=0.01
# for batchnorm: DATAGRAD = 0.001, LAMBDA=0.001

NAME=mnist_1_new
mkdir -p couts/

D=1
WD=0.0005
TRAIN=2000
LRD=piecewise
NET=lenet

for C in `seq 1 1 10`
do
    echo ITER $C
    COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN"

    # bn
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DO_BATCHNORM=True > couts/$NAME.bn_unreg_${C}.cout 2> couts/$NAME.bn_unreg_${C}.cerr
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=0.001 --DO_BATCHNORM=True > couts/$NAME.bn_spect_${C}.cout 2> couts/$NAME.bn_spect_${C}.cerr
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=0.001 --DO_BATCHNORM=True > couts/$NAME.bn_dg_${C}.cout 2> couts/$NAME.bn_dg_${C}.cerr

    # unreg
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DO_BATCHNORM=False --DROPOUT_KEEP_PROB=1 > couts/$NAME.unreg_${C}.cout 2> couts/$NAME.unreg_${C}.cerr
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=0.01 --DO_BATCHNORM=False --DROPOUT_KEEP_PROB=1 > couts/$NAME.spect_${C}.cout 2> couts/$NAME.spect_${C}.cerr
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=50 --DO_BATCHNORM=False --DROPOUT_KEEP_PROB=1 > couts/$NAME.bn_dg_${C}.cout 2> couts/$NAME.dg_${C}.cerr

    # dropout
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DO_BATCHNORM=False > couts/$NAME.do_unreg_${C}.cout 2> couts/$NAME.do_unreg_${C}.cerr
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=0.01 --DO_BATCHNORM=False > couts/$NAME.do_spect_${C}.cout 2> couts/$NAME.do_spect_${C}.cerr
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=50 --DO_BATCHNORM=False > couts/$NAME.do_dg_${C}.cout 2> couts/$NAME.do_dg_${C}.cerr	
done

