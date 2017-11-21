# Compare
# - SpectReg: approximating Frob of Jac of logits with random projection
# - JacReg: directly minimizing the Frob of jac of probs
# - Frob: directly minimizing the Frob of jac of logits
# - Onehot
# - Random Onehot
# - DataGrad

NAME=mnist_6
mkdir -p couts/
echo $NAME

D=1
WD=0.0005
LRD=piecewise
NET=lenet
TRAIN=2000
LAMBDA=0.03
JAC_LAMBDA=1
ONEHOT_LAMBDA=0.3
FROB_LAMBDA=0.03
DATAGRAD=50

for C in `seq 1 1 10`
do
    echo ITER $C
    COMMON_ARGS="--MEMORY_SHARE=0.95 --RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN"

    # Datagrad
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --DATAGRAD=$DATAGRAD > couts/$NAME.DataGrad_DG_${DATAGRAD}_${C}.cout 2> couts/$NAME.DataGrad_DG_${DATAGRAD}_${C}.cerr
    
    # Onehot
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$ONEHOT_LAMBDA --COMBINE_OUTPUTS_MODE=onehot > couts/$NAME.Onehot_LAMBDA_${ONEHOT_LAMBDA}_${C}.cout 2> couts/$NAME.Onehot_LAMBDA_${ONEHOT_LAMBDA}_${C}.cerr

    # Random Onehot
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$ONEHOT_LAMBDA --COMBINE_OUTPUTS_MODE=random_onehot > couts/$NAME.RandomOnehot_LAMBDA_${ONEHOT_LAMBDA}_${C}.cout 2> couts/$NAME.RandomOnehot_LAMBDA_${ONEHOT_LAMBDA}_${C}.cerr
    
    # SpectReg
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA > couts/$NAME.SpectReg_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.SpectReg_LAMBDA_${LAMBDA}_${C}.cerr

    # JacReg
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$JAC_LAMBDA --COMBINE_OUTPUTS_FOR_SLOPES=False --COMBINE_OUTPUTS_MODE=softmax > couts/$NAME.JacReg_LAMBDA_${JAC_LAMBDA}_${C}.cout 2> couts/$NAME.JacReg_LAMBDA_${JAC_LAMBDA}_${C}.cerr
    
    # minimizing the Frob of the Jacobian directly
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$FROB_LAMBDA --COMBINE_OUTPUTS_FOR_SLOPES=False > couts/$NAME.Jac_LAMBDA_${FROB_LAMBDA}_${C}.cout 2> couts/$NAME.Jac_LAMBDA_${FROB_LAMBDA}_${C}.cerr
    
done

echo "DONE"
