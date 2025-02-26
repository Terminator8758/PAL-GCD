CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'cifar100' \
    --eval_funcs 'v2' \
    --lr 0.01 \
    --memax_weight 4 \
    --association_interval 1 \
    --thresh 0.6 \
    --outlier_thresh 100 \
    --unlabeled_sampling 'True' \
    --sample_ratio 0.3 \
    --two_stage_joint_train 'False' \
    --memory_detach 'True'


