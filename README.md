# BabyWalk: Going Farther in Vision-and-Language Navigationby Taking Baby Steps
<img src="teaser/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This is the PyTorch implementation of our paper:

**BabyWalk: Going Farther in Vision-and-Language Navigationby Taking Baby Steps**<br>
Wang Zhu*, Hexiang Hu*, Jiacheng Chen, Zhiwei Deng, Vihan Jain, Eugene Ie, Fei Sha 
xxx (ACL), 2020

[arXiv] [[GitHub](https://github.com/Sha-Lab/babywalk)]] [Project]


## Installation

1. Install Python 3.7 (Anaconda recommended: https://www.anaconda.com/distribution/).
2. Install PyTorch following the instructions on https://pytorch.org/ (we used PyTorch 1.1.0 in our experiments).
3. Download this repository or clone with Git, and then enter the root directory of the repository:  
```
git clone https://github.com/Sha-Lab/babywalk
cd babywalk
```
4. Check the installation of required packages in requirement.txt.
5. Download and preprocess the data
```
chmod +x download.sh
./download.sh
```
After this step, 
+ `simulator/resnet_feature/` should contain `ResNet-152-imagenet.tsv`. 
+ `src/vocab/vocab_data` should contain vocabulary and its glove embedding files `train_vocab.txt` and `train_glove.npy`.
+ `tasks/` should contain `R2R`, `R4R`, `R6R`, `R8R`, `R2T8`, each which a data folder in it containing training/evaluation data.

## Training and evaluation
Here we take training on R2R as an example, using BABYWALK.

### Warmup with IL
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_follower.py \
    --split_postfix "_landmark" \
    --task_name R2R \
    --n_iters 30000 \
    --model_name "follower_bbw" \
    --il_mode "landmark_split" \
    --one_by_one \
    --one_by_one_mode "landmark" \ 
    --history                                                       --log_every 5000
```

### Training with CRL
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_follower.py \
    --split_postfix "_landmark" \
    --task_name R2R \
    --n_iters 50000 \
    --model_name "follower_bbw_crl" \
    --one_by_one \
    --one_by_one_mode "landmark" \
    --history \
    --log_every 1000 \
    --reward \
    --reward_type "cls" \
    --batch_size 64 \
    --curriculum_rl \
    --max_curriculum 4 \
    --follower_prefix "tasks/R2R/follower/snapshots/follower_bbw_sample_train_iter_30000"
```

## Other baselines
Here we take training on R2R as an example, using Reinforced Cross-modal Matching.

### Reinforced Cross-modal Matching
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_follower.py \
    --task_name R2R \
    --n_iters 50000 \
    --model_name "follower_sf_aug" \
    --max_ins_len 200 \
    --max_steps 20
    --add_augment
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_follower.py \
    --task_name R2R \
    --n_iters 20000 \
    --model_name "follower_rcm_cls" \
    --max_ins_len 200 \
    --max_steps 20 \
    --reward \
    --reward_type "cls" \
    --batch_size 64 \
    --follower_prefix "tasks/R2R/follower/snapshots/follower_sf_aug_sample_train-literal_speaker_data_augmentation_iter_50000"
```

### Evaluation
Here we take model trained on R2R, using BABYWALK as an example. <br>
evaluate on the validation unseen data of Room 2-to-8
```
CUDA_VISIBLE_DEVICES=0 python src/val_follower.py \
    --task_name R2T8 \
    --split_postfix "_landmark" \
    --one_by_one \
    --one_by_one_mode "landmark" \
    --model_name "follower_bbw"
    --history \
    --follower_prefix "tasks/R2R/follower/snapshots/best_model"
```

evaluate on the validation seen / unseen data of R$x$R ($x=2,4,6,8$)
```
CUDA_VISIBLE_DEVICES=0 python src/val_follower.py --blank --adj_feature --cls_eval \
#--follower_prefix "tasks/R2R/follower/snapshots/follower_landmark_edit_context_nested_reward_after_tune_sample_train_iter_600_val_unseen-sr=0.431" \
#--task_name R2R --model_name "follower_nbs" --split_postfix "_dpplus_edit_landmark_headings" --history --nested --one_by_one --one_by_one_mode "landmark"

```
evaluate on the test data of R2R
```
CUDA_VISIBLE_DEVICES=0 python src/val_follower.py \
    --task_name R2R \
    --split_postfix "_landmark" \
    --one_by_one \
    --one_by_one_mode "landmark" \
    --model_name "follower_bbw_test" \
    --history \
    --use_test \
    --follower_prefix "tasks/R2R/follower/snapshots/best_model"
```

### Download reported models in our paper
```
chmod +x download_model.sh
./download_model.sh
```