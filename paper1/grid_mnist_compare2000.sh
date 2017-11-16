# Compare datagrad, entReg, GP on trainsize=2000
# lenet DATAGRAD=50, LAMBDA=0.03, EP=0.01
# lenettuned DATAGRAD=50, LAMBDA=0.01, EP=0.03

NAME=mnist_compare2000
mkdir -p couts/
echo $NAME

D=1
WD=0.0005
LRD=piecewise
TRAIN=2000

NETS=( lenet lenettuned )
LAMBDAS=( 0.03 0.01 )
EPS=( 0.01 0.03 )
DATAGRAD=50


for C in `seq 1 1 10`
do
    echo ITER $C
	for i in 0 1
	do
		NET=${NETS[i]}
		echo " NET " $NET
		LAMBDA=${LAMBDAS[i]}
		EP=${EPS[i]}

		for D in 0 $DATAGRAD
		do
			echo "  D " $D
			for L in 0 $LAMBDA
			do
				echo "   L " $L
				for E in 0 $EP
				do
					echo "    E " $E
					
					COMMON_ARGS="--RANDOM_SEED=$C --LEARNING_RATE_DECAY=$LRD --WEIGHT_DECAY=$WD --DISC_TYPE=$NET --TRAIN_DATASET_SIZE=$TRAIN --DATAGRAD=$D --LAMBDA=$L --ENTROPY_PENALTY=$E"
					FILENAME=couts/$NAME.do_DG_${D}_EP_${E}_LAMBDA_${L}_NET_${NET}_${C}
					CUDA_VISIBLE_DEVICES=$D python classifier.py $COMMON_ARGS > $FILENAME.cout 2> $FILENAME.cerr
				done
			done
		done
	done
done
