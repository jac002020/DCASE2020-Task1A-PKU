#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='/home/cdd/code2/dcase2020/task1a'

# You need to modify this path to your workspace to store features and models
WORKSPACE='/home/cdd/code2/dcase2020_task1/workspace'

# Hyper-parameters
GPU_ID=2
#MODEL_TYPE='Logmel_Cnn'
#MODEL_TYPE='Cqt_Cnn'
#MODEL_TYPE='Gamm_Cnn'
MODEL_TYPE='Mfcc_Cnn'
#MODEL_TYPE='Ensemble_CNN'
#MODEL_TYPE='Logmel_Res38'
#MODEL_TYPE='Logmel_Wavegram_Logmel_Cnn14'
#MODEL_TYPE='Logmel_Cnn14'
#MODEL_TYPE='Logmel_Cnn10'
#MODEL_TYPE='Logmel_MobileNetV2'
#MODEL_TYPE='Logmel_MobileNetV1'
#MODEL_TYPE='Logmel_Wavegram_Cnn14'
#MODEL_TYPE='Ensemble_CNN2'
#MODEL_TYPE='Logmel_MultiFrebands_CNN'
#MODEL_TYPE='Logmel_SubFrebands_CNN'
#MODEL_TYPE='Cqt_SubFrebands_CNN'
#MODEL_TYPE='Gamm_SubFrebands_CNN'
#MODEL_TYPE='Mfcc_SubFrebands_CNN'
#MODEL_TYPE='Logmel_MultiFrames_CNN'
#MODEL_TYPE='Cqt_MultiFrames_CNN'
#MODEL_TYPE='Gamm_MultiFrames_CNN'
#MODEL_TYPE='Mfcc_MultiFrames_CNN'
#MODEL_TYPE='Ensemble_CNN3'
#MODEL_TYPE='Ensemble_CNN5'
#MODEL_TYPE='Ensemble_CNN6'
#MODEL_TYPE='Ensemble_CNN4'
#MODEL_TYPE='Ensemble_Models'


#### Origanl Train (Other Models)
BATCH_SIZE=64
ITE_TRAIN=24000
ITE_EVA=23800
ITE_STORE=23800
FIXED=False
FINETUNE=False

```
#### Train last layer (PANNS Models)
BATCH_SIZE=32
ITE_TRAIN=2000
ITE_EVA=1800
ITE_STORE=1800
FIXED=True
FINETUNE=False
```
```
#### Finetune (PANNS Models)
BATCH_SIZE=24
ITE_TRAIN=12000
ITE_EVA=10000
ITE_STORE=10000
FIXED=False
FINETUNE=True
```
############ Train and validate on development dataset ############
# Calculate feature
#python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='a' --data_type='development' --workspace=$WORKSPACE

# Calculate scalar
#python utils/features.py calculate_scalar --subtask='a' --data_type='development' --workspace=$WORKSPACE

# Subtask A
#CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --ite_train=$ITE_TRAIN --ite_eva=$ITE_EVA --ite_store=$ITE_STORE --fixed=$FIXED --finetune=$FINETUNE --cuda

#CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --iteration=8000 --batch_size=$BATCH_SIZE --cuda

# Train on full data
CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main_eva.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --ite_train=$ITE_TRAIN --ite_eva=$ITE_EVA --ite_store=$ITE_STORE --fixed=$FIXED --finetune=$FINETUNE --cuda

# # Plot statistics
# python utils/plot_results.py --workspace=$WORKSPACE --subtask=a

# ############ Train on full data without validation, inference on leaderboard and evaluation data ############

# # Extract features for leaderboard data
# python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='a' --data_type='leaderboard' --workspace=$WORKSPACE
# python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='b' --data_type='leaderboard' --workspace=$WORKSPACE
# python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='c' --data_type='leaderboard' --workspace=$WORKSPACE

# # Train on full data
# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='development' --holdout_fold='none' --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='b' --data_type='development' --holdout_fold='none' --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='c' --data_type='development' --holdout_fold='none' --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --cuda

# # Inference on leaderboard data
# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_evaluation --workspace=$WORKSPACE --subtask='a' --data_type='leaderboard' --model_type=$MODEL_TYPE --iteration=5000 --batch_size=$BATCH_SIZE --cuda

# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_evaluation --workspace=$WORKSPACE --subtask='b' --data_type='leaderboard' --model_type=$MODEL_TYPE --iteration=5000 --batch_size=$BATCH_SIZE --cuda

# CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_evaluation --workspace=$WORKSPACE --subtask='c' --data_type='leaderboard' --model_type=$MODEL_TYPE --iteration=5000 --batch_size=$BATCH_SIZE --cuda