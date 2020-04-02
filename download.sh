#!/usr/bin/env bash

# vocab data
mkdir -p src/vocab/vocab_data
wget https://www.dropbox.com/s/r71i31xpm1zy3oy/sub_train_vocab.txt?dl=0 -O src/vocab/vocab_data/sub_train_vocab.txt
wget https://www.dropbox.com/s/xqt6et0i1g41t88/train_glove.npy?dl=0 -O src/vocab/vocab_data/train_glove.npy
wget https://www.dropbox.com/s/l7dee5fls07t9q0/train_vocab.txt?dl=0 -O src/vocab/vocab_data/train_vocab.txt
wget https://www.dropbox.com/s/cjapgv3rpxrq1ie/trainval_glove.npy?dl=0 -O src/vocab/vocab_data/trainval_glove.npy
wget https://www.dropbox.com/s/3s2plada1vttxuv/trainval_vocab.txt?dl=0 -O src/vocab/vocab_data/trainval_vocab.txt

# resnet feature
mkdir -p simulator/resnet_feature/
wget https://www.dropbox.com/s/715bbj8yjz32ekf/ResNet-152-imagenet.zip?dl=1 -O simulator/resnet_feature/ResNet-152-imagenet.zip
unzip simulator/resnet_feature/ResNet-152-imagenet.zip -d simulator/resnet_feature

# training/eval data
mkdir -p tasks/R2R/data
mkdir -p tasks/R4R/data
mkdir -p tasks/R6R/data
mkdir -p tasks/R8R/data
mkdir -p tasks/R2T8/data
wget https://www.dropbox.com/s/2v3f72vpoj53r6d/R2R_data.zip?dl=0 -O tasks/R2R/data/R2R_data.zip
wget https://www.dropbox.com/s/7n7ptzkjr601dq9/R4R_data.zip?dl=0 -O tasks/R4R/data/R4R_data.zip
wget https://www.dropbox.com/s/bjqwu9tn0t6f50r/R6R_data.zip?dl=0 -O tasks/R6R/data/R6R_data.zip
wget https://www.dropbox.com/s/kdid25goi88sgxo/R8R_data.zip?dl=0 -O tasks/R8R/data/R8R_data.zip
wget https://www.dropbox.com/s/aswlh36v68x3al0/R2T8_data.zip?dl=0 -O tasks/R2T8/data/R2T8_data.zip
unzip tasks/R2R/data/R2R_data.zip -d tasks/R2R/data
unzip tasks/R4R/data/R4R_data.zip -d tasks/R4R/data
unzip tasks/R6R/data/R6R_data.zip -d tasks/R6R/data
unzip tasks/R8R/data/R8R_data.zip -d tasks/R8R/data
unzip tasks/R2T8/data/R2T8_data.zip -d tasks/R2T8/data