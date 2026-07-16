for model in BiLSTM CNN CNN_BiLSTM; do
    for fold in 1 2 3 4 5; do
        python 06_train_tic_segmentation.py \
            --fold "$fold" \
            --model-name "$model" \
            --split-by session \
            --feat-name MFCC
    done
done
