import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter
from util.cluster_and_log_utils import log_accs_from_preds
from config import dino_pretrain_path
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups

import sys
import os
from models import vision_transformer as vits
from util.logging import Logger
from util.osutils import str2bool

from association.semi_association import semi_association
from association.proxy_memory import ProxyMemory
from data.data_utils import GCDClassUniformlySampler, IterLoader


def train(student, train_loader, unlabelled_train_loader, propagate_loader, args, train_index_mapper=None, stage=1):

    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    cluster_criterion = DistillLoss(args.warmup_teacher_temp_epochs, args.epochs, args.n_views, args.warmup_teacher_temp, args.teacher_temp)
    
    gcd_memory = ProxyMemory(temp=args.memory_temp, momentum=args.memory_mu).cuda()

    memory_weight = 1 if stage==1 else 0.1
    print(f'Stage {stage} memory_weight= {memory_weight}')

    train_epochs = args.epochs
    num_iters = args.num_iters

    if (args.dataset_name in ['cifar100', 'imagenet_100']) and (stage==1):
        train_epochs = 30
        if args.unlabeled_sampling:
            num_iters = int(1.0*(args.label_len + args.sample_ratio*args.unlabelled_len)/args.batch_size)


    for epoch in range(train_epochs):

        loss_record = AverageMeter()
        
        if (epoch >= args.association_start_epoch) and (epoch % args.association_interval==0):

            proxy_centers, proxy_label, img_proxy_index, cluster_img_dict, all_proxy_mask = semi_association(
                                          student, propagate_loader, thresh=args.thresh, rerank=True, associate_type='thresholding',
                                          labeled_cls_num=args.num_labeled_classes, outlier_thresh=args.outlier_thresh, assign_outliers=True,
                                          unlabeled_sampling=args.unlabeled_sampling, sample_ratio=args.sample_ratio, epoch=epoch)

            gcd_memory.proxy_memory = proxy_centers
            gcd_memory.all_proxy_label = proxy_label
            gcd_memory.img_proxy_index = img_proxy_index
            gcd_memory.all_proxy_mask = all_proxy_mask

            ot_num = len(torch.nonzero(img_proxy_index==-1).squeeze(-1))
            print('After semi-association: per-image proxy index outlier num= ', ot_num)

            if (train_index_mapper is not None) and (stage == 1):
                print('Re-initializing PK-sampler and train loader...')
                pk_sampler = GCDClassUniformlySampler(train_index_mapper, cluster_img_dict, k=args.num_instances)
                train_loader = IterLoader(DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                            shuffle=False, sampler=pk_sampler, drop_last=True, pin_memory=True), length=num_iters)
                train_loader.new_epoch()

        student.train()

        for batch_idx in range(len(train_loader)):

            images, class_labels, uq_idxs, mask_lab = train_loader.next()
            
            mask_lab = mask_lab[:, 0]
            class_labels, mask_lab, uq_idxs = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool(), uq_idxs.cuda(non_blocking=True)
            images = torch.cat(images, dim=0).cuda(non_blocking=True)
            #print('batch GT label= ', class_labels.cpu().detach())
            #print('batch images shape= ', images.shape)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out, student_ori_feat = student(images)

                cls_loss = torch.tensor(0.)
                cluster_loss = torch.tensor(0.)
                sup_con_loss = torch.tensor(0.)
                contrastive_loss = torch.tensor(0.)
                memory_loss = torch.tensor(0.).cuda()

                # SimGCD loss
                if stage == 2:
                    teacher_out = student_out.detach()
                    ### clustering, sup
                    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                    ### clustering, unsup
                    cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                    cluster_loss += args.memax_weight * me_max_loss

                    # represent learning, unsup
                    contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                    # representation learning, sup
                    student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)  # shape=(n,2,768)
                    student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                    sup_con_labels = class_labels[mask_lab]
                    sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                # non-parametric contrastive loss using memory prototypes
                if epoch >= args.association_start_epoch:
                    uq_idxs = torch.cat((uq_idxs, uq_idxs))  # to match the image output
                    val_inds = torch.nonzero(img_proxy_index[uq_idxs]>=0).squeeze(-1)
                    if len(val_inds) > 0:
                        memory_loss += memory_weight * gcd_memory(student_ori_feat[val_inds], uq_idxs[val_inds], detach_feat=args.memory_detach)

                pstr = ''
                pstr += f' cls_loss: {cls_loss.item():.4f} '
                pstr += f' cluster_loss: {cluster_loss.item():.4f} '
                pstr += f' sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f' unsup_loss: {contrastive_loss.item():.4f} '
                pstr += f' memory_con_loss: {memory_loss.item():.4f} '

                loss = torch.tensor(0.).cuda()
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                loss += memory_loss
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t loss: {:.5f}\t {}'.format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        print('Train Epoch: {}   Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        
        if (stage == 2) and (epoch % 5 == 0):
            print('Testing (classifier logit based) on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
            print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        save_name = args.model_path[:-3] + '_stage_' + str(stage) + '.pt'
        torch.save(save_dict, save_name)
        print("model saved to {}.".format(save_name))


def test(model, test_loader, epoch, save_name, args):

    model.eval()
    print('loader size= ', len(test_loader))
    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits, _ = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args)
    return all_acc, old_acc, new_acc



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GCD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2',])

    parser.add_argument('--dataset_name', type=str, default='xxx', help='options: cifar100, imagenet_100, cub, scars, aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)  # labeled instance ratio
    parser.add_argument('--old_class_ratio', type=float, default=0.5)    # known class ratio among all classes
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    #parser.add_argument('--dino_pretrain_path', type=str, default='../SimGCD-main/pretrained_models/dino_vitbase16_pretrain.pth')

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=1)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=20, type=int)

    # association-related setting:
    parser.add_argument('--association_interval', type=int, default=1)
    parser.add_argument('--association_start_epoch', type=int, default=0)
    parser.add_argument('--num_instances', type=int, default=16)

    parser.add_argument('--thresh', type=float, default=0.35)
    parser.add_argument('--outlier_thresh', type=int, default=10)
    parser.add_argument('--unlabeled_sampling', type=str2bool, default=False)
    parser.add_argument('--sample_ratio', type=float, default=0.5)

    parser.add_argument('--memory_temp', type=float, default=0.05)
    parser.add_argument('--memory_mu', type=float, default=0.2)
    parser.add_argument('--memory_detach', type=str2bool, default=False)

    parser.add_argument('--two_stage_joint_train', type=str2bool, default=True)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')

    #print('check after parser.parse_args(): args= ', args)
    args.log_dir = args.dataset_name+ '_logs/'
    args.log_file_name = args.log_dir + 'train.txt'
    sys.stdout = Logger(args.log_file_name)
    args.model_dir = 'trained_models'
    args.model_path = os.path.join(args.model_dir, args.dataset_name+'_dino_pretrain_PAL_GCD_model.pt')

    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    print('==================> args= ', args)

    torch.backends.cudnn.benchmark = True


    # ----------------------
    # BASE MODEL
    # ----------------------
    ### dinov2_vitb14
    #backbone = torch.hub.load('../dinov2', 'dinov2_vitb14', source='local', pretrained=False)
    #backbone.load_state_dict(torch.load(dino_pretrain_path))    

    ### dino_vitb16
    backbone = vits.__dict__['vit_base']()
    state_dict = torch.load(dino_pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.gt_num_classes = args.num_labeled_classes + args.num_unlabeled_classes   # using GT-class number to define classifier length

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                print('Vit-B trainable block name= ', name)
                m.requires_grad = True
    
    print('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, trainset_propagate, datasets = \
            get_datasets(args.dataset_name, train_transform, test_transform, args, return_propagate_loader=True) 

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    args.label_len = label_len
    args.unlabelled_len = unlabelled_len
    print('  train dataset info: labeled instance number= {}, unlabeled instance number= {}'.format(label_len, unlabelled_len))
    print('                      labeled class number= {}, unlabeled class number= {}'.format(args.num_labeled_classes, args.num_unlabeled_classes))


    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # PK sampler:
    train_img_num = len(datasets['train_labelled'])+len(datasets['train_unlabelled'])
    labeled_train_num = len(datasets['train_labelled'])
    train_index_mapper = torch.zeros(train_img_num).long()
    for i, idx in enumerate(datasets['train_labelled'].uq_idxs):
        train_index_mapper[idx] = i
    for i, idx in enumerate(datasets['train_unlabelled'].uq_idxs):
        train_index_mapper[idx] = i+labeled_train_num

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    train_loader = IterLoader(train_loader, length=int(len(train_dataset)/args.batch_size))
    train_loader.new_epoch()

    args.num_iters = len(train_loader)

    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    if trainset_propagate is not None:
        train_loader_propagate = DataLoader(trainset_propagate, num_workers=args.num_workers,
                                            batch_size=256, shuffle=False, pin_memory=False)  # new loader for association
        print('train_loader_propagate: length= ', len(train_loader_propagate))
    else:
        train_loader_propagate = None

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.gt_num_classes, nlayers=args.num_mlp_layers, bn_after_x=True)
    model = nn.Sequential(backbone, projector)
    model.cuda()
    model = nn.DataParallel(model)

    # stage-1 train:
    print('\n ===> Stage-1 non-parametric training... \n')
    train(model, None, test_loader_unlabelled, train_loader_propagate, args, train_index_mapper, stage=1)
  

    if args.two_stage_joint_train:
        print('\n ===> Stage-2 joint non-parametric and parametric classifier training... \n')
        # stage-2 train:
        args.epochs = 200
        args.lr = 0.1  # reset learning rate for stage-2 joint train
        args.memory_detach = True

        if args.dataset_name=='cifar100' or args.dataset_name=='herbarium_19':
            args.association_interval = 200
        elif args.dataset_name=='imagenet_100':
            args.association_interval = 200  # 5
        else:
            args.association_interval = 1

        train(model, train_loader, test_loader_unlabelled, train_loader_propagate, args, train_index_mapper, stage=2)



