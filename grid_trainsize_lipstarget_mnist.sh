NAME=trainsize_lipstarget_mnist
VAR1=TRAIN_DATASET_SIZE
VAR2=LIPSCHITZ_TARGET

mkdir -p couts/
cp classifier.py couts/$NAME.classifier.py

for VAL2 in 0.00001 0.0001 0.001 0.01 0.1 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 20.0 50.0
do
    for VAL1 in `seq 100 100 900` `seq 1000 1000 10000`
    do
        CUDA_VISIBLE_DEVICES=1 python classifier.py --ITERS=10000 --DATASET=mnist --LAMBDA=0.0001 --LEARNING_RATE=0.0001 --DO_BATCHNORM=0 --DISC_TYPE=conv --$VAR1=$VAL1 --$VAR2=$VAL2 > couts/$NAME.${VAR1}_${VAL1}.${VAR2}_${VAL2}.cout 2> couts/$NAME.${VAR1}_${VAL1}.${VAR2}_${VAL2}.cerr
    done
done
