for model in CNN; do
    for feat in MFCC WavLM MelSpectrogram Spectrogram;do
    for fold in 1 2 3 4 5; do
        python 06_train_tic_segmentation.py \
            --fold "$fold" \
            --model-name "$model" \
            --split-by session \
            --feat-name $feat
        done
    done
done