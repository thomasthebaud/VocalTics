for model in TCNN ResNet34;do
    for f in 1 2 3 4 5;do 
        python 05_train_tic_detection.py \
        --fold $f \
        --model-name $model \
        --split-by session \
        --feat-name MFCC
    done
done