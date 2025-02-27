#!/bin/bash

RAW_DATA_DIR='/scratch/jayakumar/SC22D036/resampled/'
PREPROC_DIR='/scratch/jayakumar/SC22D036/preprocessed/egnn/'
SAVE_DIR='/home/jayakumar/SC22D036/ramzan/s4gnn/save/wpli/'
BATCH_SIZE=7

python train.py \
    --dataset 'tuh' \
    --raw_data_dir $RAW_DATA_DIR \
    --preproc_dir $PREPROC_DIR \
    --max_seq_len 60 \
    --num_nodes 19 \
    --input_dim 1 \
    --output_dim 1 \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --num_workers 8 \
    --adj_mat_dir 'data/eeg_electrode_graph/adj_mx_3d.pkl' \
    --model_name 's4_gnn' \
    --graph_learn_metric "self_attention" \
    --dropout 0.1 \
    --g_conv 'gine' \
    --num_gcn_layers 3 \
    --hidden_dim 128 \
    --num_temporal_layers 4 \
    --state_dim 64 \
    --bidirectional False \
    --temporal_model 's4' \
    --temporal_pool 'mean' \
    --resolution 2000 \
    --graph_pool 'max' \
    --activation_fn 'leaky_relu' \
    --thresh 0.1 \
    --use_prior True \
    --knn 2 \
    --save_dir $SAVE_DIR \
    --metric_name 'auroc' \
    --eval_metrics 'acc' 'auroc' 'F1' 'precision' 'recall' \
    --metric_avg 'binary' \
    --lr_init 8e-4 \
    --l2_wd 5e-3 \
    --num_epochs 100 \
    --scheduler timm_cosine \
    --t_initial 100 \
    --warmup_t 5 \
    --optimizer adamw \
    --do_train True \
    --balanced_sampling True \
    --accumulate_grad_batches 1 \
    --fn_connectivity 'spearmanr' \
    --gpus 1 \