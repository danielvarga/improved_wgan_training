# Compare SpectReg (approximating Frob of Jac with random projection) with directly minimizing the Frob of Jac

NAME=mnist_6
mkdir -p couts/
echo $NAME

D=1
WD=0.0005
LRD=piecewise
NET=lenet
TRAIN=2000
LAMBDA=0.03
JAC_LAMBDAS=( 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 )

for C in `seq 1 1 10`
do
    echo ITER $C
    COMMON_ARGS="--MEMORY_SHARE=0.95 --RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN"

    # SpectReg
    CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$LAMBDA > couts/$NAME.SpectReg_LAMBDA_${LAMBDA}_${C}.cout 2> couts/$NAME.SpectReg_LAMBDA_${LAMBDA}_${C}.cerr

    # minimizing the Jacobian directly
    for JAC_LAMBDA in JAC_LAMBDAS
    do
	
	CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS --LAMBDA=$JAC_LAMBDA --COMBINE_OUTPUTS_FOR_SLOPES=False > couts/$NAME.Jac_LAMBDA_${JAC_LAMBDA}_${C}.cout 2> couts/$NAME.Jac_LAMBDA_${JAC_LAMBDA}_${C}.cerr
	
    done
done

echo "DONE"
