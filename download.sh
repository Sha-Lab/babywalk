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
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=1HjEH3EQt-aHSjolg0VnX_YF1UEHiXLfT" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > simulator/resnet_feature/ResNet-152-imagenet.zip
unzip simulator/resnet_feature/ResNet-152-imagenet.zip -d simulator/resnet_feature

# adjacency dict
wget https://www.dropbox.com/s/6a076293c3o77gi/total_adj_list.json?dl=0 -O simulator/total_adj_list.json

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

# download speaker model
mkdir -p tasks/R2R/speaker/snapshots
mkdir -p tasks/R4R/speaker/snapshots
mkdir -p tasks/R6R/speaker/snapshots
mkdir -p tasks/R8R/speaker/snapshots
wget https://www.dropbox.com/s/65z90zktd7w6dtz/speaker.zip?dl=0 -O tasks/R2R/speaker/snapshots/speaker.zip
wget https://www.dropbox.com/s/q223j0vn1ofd89z/speaker.zip?dl=0 -O tasks/R4R/speaker/snapshots/speaker.zip
unzip tasks/R2R/speaker/snapshots/speaker.zip -d tasks/R2R/speaker/snapshots
unzip tasks/R4R/speaker/snapshots/speaker.zip -d tasks/R4R/speaker/snapshots
cp tasks/R4R/speaker/snapshots/* tasks/R6R/speaker/snapshots
cp tasks/R6R/speaker/snapshots/* tasks/R8R/speaker/snapshots