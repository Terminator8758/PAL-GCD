CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'herbarium_19' \
    --eval_funcs 'v2' \
    --lr 0.01 \
    --epochs 100 \
    --memax_weight 1 \
    --association_interval 1 \
    --thresh 0.6 \
    --outlier_thresh 4 \
    --unlabeled_sampling 'False' \
    --two_stage_joint_train 'False'


