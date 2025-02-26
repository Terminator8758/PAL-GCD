import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.cluster import KMeans
#from project_utils.cluster_utils import my_mixed_eval, cluster_acc

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from .faster_mix_k_means_pytorch import K_Means
from .cluster_and_log_utils import log_accs_from_preds



def test_kmeans_semi_sup_with_pred_centers(model, merge_test_loader, args, K=None, mode='train', intra_cluster_kmeans=False):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """
    device = torch.device('cuda:0')
    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to Old classes
    all_img_idxs = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, img_idxs, mask_lab_) in enumerate(tqdm(merge_test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, _, feats = model(images)

            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                            else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())
            all_img_idxs = np.append(all_img_idxs, img_idxs.cpu().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)
    l_feats = all_feats[mask_lab]       # Get labelled set
    u_feats = all_feats[~mask_lab]      # Get unlabelled set
    l_targets = targets[mask_lab]       # Get labelled targets
    u_targets = targets[~mask_lab]       # Get unlabelled targets

    print('Fitting Semi-Supervised K-Means...')
    kmeans = K_Means(k=K, tolerance=1e-4, max_iterations=10, init='k-means++', n_init=10, random_state=None,
                           n_jobs=None, pairwise_batch_size=1024, mode=None)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    kmeans.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeans.labels_.cpu().numpy()
    u_targets = u_targets.cpu().numpy()

    # -----------------------
    # EVALUATE
    # -----------------------
    # Get preds corresponding to unlabelled set
    preds = all_preds[~mask_lab]

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)

    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask, eval_funcs=args.eval_funcs,
                                                        save_name='SS-K-Means Train ACC Unlabelled', print_output=True)


    # ============= compute proxy memory, proxy_label and img_proxy_index
    cnt = 0
    proxy_memory, proxy_label = [], []
    all_img_idxs = torch.from_numpy(all_img_idxs)
    all_feats = torch.from_numpy(all_feats)
    all_preds = torch.from_numpy(all_preds)
    img_proxy_index = -1 * torch.ones(all_img_idxs.shape).long()
    kmeans_algo = KMeans(n_clusters=2, random_state=10, n_init='auto')

    for lbl in torch.unique(all_preds):
        inds = torch.nonzero(all_preds == lbl).squeeze(-1)
        if intra_cluster_kmeans:

            km = kmeans_algo.fit(all_feats[inds])
            intra_label = km.labels_
            sc_1_ind = inds[intra_label == 0]  # sub-cluster indexes
            sc_2_ind = inds[intra_label == 1]

            proxy_memory.append(all_feats[sc_1_ind].mean(0))
            proxy_memory.append(all_feats[sc_2_ind].mean(0))
            proxy_label.append(lbl)
            proxy_label.append(lbl)
            img_proxy_index[all_img_idxs[sc_1_ind]] = cnt
            img_proxy_index[all_img_idxs[sc_2_ind]] = cnt + 1
            cnt += 2

        else:
            proxy_memory.append(all_feats[inds].mean(0))
            proxy_label.append(lbl)
            img_proxy_index[all_img_idxs[inds]] = cnt
            cnt += 1
    
    proxy_memory = torch.vstack(proxy_memory)
    proxy_memory = F.normalize(proxy_memory.detach(), dim=1).detach().cuda()
    proxy_label = torch.tensor(proxy_label).long().cuda()  # cluster label of each proxy in proxy_memory
    img_proxy_index = img_proxy_index.cuda()

    return all_acc, old_acc, new_acc, proxy_memory, proxy_label, img_proxy_index




def test_kmeans_semi_sup(model, merge_test_loader, args, K=None, mode='train'):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """
    device = torch.device('cuda:0')
    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to Old classes

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _, mask_lab_) in enumerate(tqdm(merge_test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, _, feats = model(images)
        
            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                            else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)
    l_feats = all_feats[mask_lab]       # Get labelled set
    u_feats = all_feats[~mask_lab]      # Get unlabelled set
    l_targets = targets[mask_lab]       # Get labelled targets
    u_targets = targets[~mask_lab]       # Get unlabelled targets

    print('Fitting Semi-Supervised K-Means...')
    kmeans = K_Means(k=K, tolerance=1e-4, max_iterations=10, init='k-means++', n_init=10, random_state=None, 
                           n_jobs=None, pairwise_batch_size=1024, mode=None)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                              x in (l_feats, u_feats, l_targets, u_targets))
    
    kmeans.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeans.labels_.cpu().numpy()
    u_targets = u_targets.cpu().numpy()

    # -----------------------
    # EVALUATE
    # -----------------------
    # Get preds corresponding to unlabelled set
    preds = all_preds[~mask_lab]

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)

    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask, eval_funcs=args.eval_funcs,
                                                        save_name='SS-K-Means Train ACC Unlabelled', print_output=True)
    return all_acc, old_acc, new_acc




@torch.no_grad()
def test_kmeans(model, test_loader, epoch, save_name, args, use_fast_Kmeans=True):

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, _, feats = model(images)
            all_feats.append(feats.detach())
            targets = np.append(targets, label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = torch.cat(all_feats, dim=0)

    if use_fast_Kmeans:
        begin = time.time()
        kmeans = K_Means(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-6, max_iterations=500, init='k-means++', 
                n_init=20, random_state=0, n_jobs=1, 
                pairwise_batch_size=None if all_feats.size(0)<args.fast_kmeans_batch_size else args.fast_kmeans_batch_size, 
                mode=None)
        kmeans.fit(all_feats)
        preds = kmeans.labels_.detach().cpu().numpy()
        end = time.time()
        print(f'time={end-begin}')
        
    else:
        begin = time.time()
        all_feats = all_feats.cpu().numpy()
        kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, verbose=0).fit(all_feats)
        end = time.time()
        print(f'time={end-begin}')
        preds = kmeans.labels_
        
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args)
    
    return all_acc, old_acc, new_acc
        



@torch.no_grad()
def test_kmeans_val(model, test_loader,
                epoch, save_name,
                args, use_fast_Kmeans=False, 
                predict_token='cls',
                return_silhouette=False,
                stage=2,
                ):
    """ KMeans validaation on @model, Inductive GNCD setting
    Args:
        return_silhouette: if true, to return unsupervised silhouette score
        stage: which training stage
    """
    model.eval()

    all_feats = []
    targets = np.array([])
    cls_mask = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    with tqdm(total=len(test_loader)) as pbar:
        for batch_idx, (images, label, _, labeled) in enumerate(test_loader):
            images = images.cuda(args.device)
            feats = forward(images, model, projection_head=None, predict_token=predict_token, mode='test')
            feats = F.normalize(feats, dim=-1)
            all_feats.append(feats.detach().cpu())
            targets = np.append(targets, label.cpu().numpy())
            cls_mask = np.append(cls_mask, np.array([True if x.item() in range(len(args.train_classes))
                                            else False for x in label]))
            mask = np.append(mask, np.array([x.item() for x in labeled]))

            pbar.update(1)
            
    mask = mask.astype(np.bool)  # labeled or unlabeled
    cls_mask = cls_mask.astype(np.bool)  # old class or new class
    
    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = torch.cat(all_feats, dim=0)
    if stage==1: ### only consider lbl data
        if use_fast_Kmeans is True:
            begin = time.time()
            all_feats = all_feats.to(args.device)
            kmeans = K_Means(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-6, max_iterations=500, init='k-means++', 
                    n_init=20, random_state=0, n_jobs=1, 
                    pairwise_batch_size=None if all_feats.size(0)<args.fast_kmeans_batch_size else args.fast_kmeans_batch_size, 
                    mode=None)
            kmeans.fit(all_feats)
            preds = kmeans.labels_.detach().cpu().numpy()
            end = time.time()
            print(f'time={end-begin}')
            all_feats = all_feats.detach().cpu().numpy()
        else:
            begin = time.time()
            all_feats = all_feats.numpy()
            kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, verbose=0).fit(all_feats)
            end = time.time()
            print(f'time={end-begin}')
            preds = kmeans.labels_
        
        lbl_acc = cluster_acc(y_true=targets[mask], y_pred=preds[mask])
        args.writer.add_scalar('Surveillance/lbl_acc', lbl_acc, epoch)
        return lbl_acc, None, None
    else: ### compute lbl_acc + total_sil score
        if use_fast_Kmeans is True:
            begin = time.time()
            all_feats = all_feats.to(args.device)
            kmeans = K_Means(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-6, max_iterations=500,
                             init='k-means++', n_init=20, random_state=0, n_jobs=1,
                             pairwise_batch_size=None if all_feats.size(0)<args.fast_kmeans_batch_size else args.fast_kmeans_batch_size,
                             mode=None)
            kmeans.fit(all_feats)
            preds = kmeans.labels_.detach().cpu().numpy()
            end = time.time()
            print(f'time={end-begin}')
            all_feats = all_feats.detach().cpu().numpy()
        else:
            begin = time.time()
            all_feats = all_feats.numpy()
            kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, verbose=0).fit(all_feats)
            end = time.time()
            print(f'time={end-begin}')
            preds = kmeans.labels_
        

        # -----------------------
        # EVALUATE
        # -----------------------
        lbl_acc = cluster_acc(y_true=targets[mask], y_pred=preds[mask])
        args.writer.add_scalar('Surveillance/lbl_acc', lbl_acc, epoch)
        unlbl_sil = silhouette_score(all_feats[~mask], preds[~mask])
        args.writer.add_scalar('Surveillance/unlbl_sil', unlbl_sil, epoch)
        
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=cls_mask,
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                        writer=args.writer)
        
        res, ratio = my_mixed_eval(targets, preds, cls_mask, all_feats)
        for k, v in res.items():
            args.writer.add_scalar(f'{save_name}-cluster/{k}', v, epoch)
            print(f'cluster/{k}={v}')
            
        return lbl_acc, unlbl_sil, res['total_sil']
