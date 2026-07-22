for model in CNN;do
    for feat in MFCC WavLM MelSpectrogram Spectrogram;do
        for f in 1 2 3 4 5;do 
            python 05_train_tic_detection.py \
            --fold $f \
            --model-name $model \
            --split-by session \
            --feat-name $feat
    done
done