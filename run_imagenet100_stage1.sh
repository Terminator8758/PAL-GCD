CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'imagenet_100' \
    --eval_funcs 'v2' \
    --lr 0.01 \
    --memax_weight 1 \
    --association_interval 5 \
    --thresh 0.6 \
    --outlier_thresh 100 \
    --unlabeled_sampling 'True' \
    --sample_ratio 0.3 \
    --two_stage_joint_train 'False' \
    --memory_detach 'True'


