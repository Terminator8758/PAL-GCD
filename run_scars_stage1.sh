CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'scars' \
    --eval_funcs 'v2' \
    --lr 0.01 \
    --memax_weight 1 \
    --association_interval 1 \
    --thresh 0.35 \
    --outlier_thresh 10 \
    --unlabeled_sampling 'False' \
    --two_stage_joint_train 'False'


