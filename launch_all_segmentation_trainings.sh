for fold in 1 2 3 4 5; do
    python 06_train_tic_segmentation.py \
        --fold "$fold" \
        --model-name CNN_BiLSTM \
        --split-by session \
        --feat-name MFCC
done
