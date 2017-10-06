if [ "$#" -le 3 ]; then
    echo "Usage: bash grid.sh EXP_NAME PARAMETER VALUE1 VALUE2 ..."
    echo "Writes to couts/NAME.PARAMETER_VALUE.cout"
    echo "For example:"
    echo "CUDA_VISIBLE_DEVICES=1 bash grid.sh test TRAIN_DATASET_SIZE `seq 2000 10000 2000`"
    exit 1
fi


NAME=$1
VAR=$2

mkdir -p couts/
for ((i=3; i<=$#; i++))
do
    VAL=${!i}
    echo $CUDA_VISIBLE_DEVICES
    python classifier.py --$VAR=$VAL > couts/$NAME.${VAR}_${VAL}.cout 2> couts/$NAME.${VAR}_${VAL}.cerr
done
