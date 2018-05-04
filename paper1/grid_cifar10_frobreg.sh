# BS has been reduces from 128 to 32 !!!
# Tune frobreg

NAME=cifar10_frobreg
mkdir -p couts/
echo $NAME

D=2
DATASET=cifar10
ITERS=50000
BS=32
LR=0.01
LRD=piecewise
WD=0.003
NET=cifarResnet
BN=True
TRAIN=2000
AUGMENTATION=False
WIDENESS=3
LAMBDA=0.003
JAC_LAMBDA=1
ONEHOT_LAMBDA=0.3
# FROB_LAMBDA=0.03
DATAGRAD=1

for C in `seq 1 1 10`
do
    echo ITER $C
    for FROB_LAMBDA in 0.003 0.01 0.03 0.1 0.3
    do
        COMMON_ARGS="--RANDOM_SEED=$C --MEMORY_SHARE=0.95 --DATASET=$DATASET --ITERS=$ITERS --BATCH_SIZE=$BS --LEARNING_RATE=$LR --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --WIDENESS=$WIDENESS --DO_BATCHNORM=$BN --TRAIN_DATASET_SIZE=$TRAIN --AUGMENTATION=$AUGMENTATION"

    # Datagrad
#    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DATAGRAD > couts/$NAME.DataGrad_DG_${DATAGRAD}_${C}.cout 2> couts/$NAME.DataGrad_DG_${DATAGRAD}_${C}.cerr
    
    # Onehot
#    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$ONEHOT_LAMBDA --COMBINE_OUTPUTS_MODE=onehot > couts/$NAME.Onehot_LAMBDA_${ONEHOT_LAMBDA}_${C}.cout 2> couts/$NAME.Onehot_LAMBDA_${ONEHOT_LAMBDA}_${C}.cerr

    # Random Onehot
#    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$ONEHOT_LAMBDA --COMBINE_OUTPUTS_MODE=random_onehot > couts/$NAME.RandomOnehot_LAMBDA_${ONEHOT_LAMBDA}_${C}.cout 2> couts/$NAME.RandomOnehot_LAMBDA_${ONEHOT_LAMBDA}_${C}.cerr

    # JacReg
#    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$JAC_LAMBDA --COMBINE_OUTPUTS_FOR_SLOPES=False --COMBINE_OUTPUTS_MODE=softmax > couts/$NAME.JacReg_LAMBDA_${JAC_LAMBDA}_${C}.cout 2> couts/$NAME.JacReg_LAMBDA_${JAC_LAMBDA}_${C}.cerr
    
    # SpectReg
#    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA > couts/$NAME.SpectReg_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.SpectReg_LAMBDA_${LAMBDA}_${C}.cerr
    
        # minimizing the Frob of the Jacobian directly
        CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$FROB_LAMBDA --COMBINE_OUTPUTS_FOR_SLOPES=False > couts/$NAME.FrobReg_LAMBDA_${FROB_LAMBDA}_${C}.cout 2> couts/$NAME.FrobReg_LAMBDA_${FROB_LAMBDA}_${C}.cerr
    
    done
done

echo "DONE"
