import torch
from torch.nn import functional as F
import numpy as np
import time
from util.cluster_and_log_utils import split_cluster_acc_v2, cluster_acc
from sklearn.cluster import KMeans
from util.faiss_rerank import compute_jaccard_distance
from util.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from association.semi_dbscan import SemiDBSCAN
 

def semi_association(model, propagate_loader, thresh=0.5, rerank=False, k1=20, k2=6, associate_type='kmeans',
                     total_class_num=0, labeled_cls_num=5, subclass_split_num=1, outlier_thresh=4, assign_outliers=True,
                     fp16=False, unlabeled_sampling=False, sample_ratio=0.5, epoch=0):

    print('==> start semi-supervised association with labeled sub-classes...')
    device = torch.device('cuda:0')
    model.eval()

    all_feats = []
    all_labels = np.array([], dtype=np.int64)
    all_img_idxs = np.array([])
    mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
    old_or_new_mask = np.array([], dtype=bool)

    for batch_idx, (images, label, img_idxs, mask_lab_) in enumerate(propagate_loader):
        images = images.to(device)
        with torch.no_grad():    
            _, _, feats = model(images)

        all_feats.append(feats.detach())  
        all_labels = np.append(all_labels, label.cpu().numpy())
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())
        all_img_idxs = np.append(all_img_idxs, img_idxs.cpu().numpy())
        old_or_new_mask = np.append(old_or_new_mask, np.array([True if x.item() in range(labeled_cls_num) else False for x in label]))

    all_feats = torch.vstack(all_feats)
    all_feats = F.normalize(all_feats, p=2, dim=1)
    print('    all train feats shape= {}, dtype= {}'.format(all_feats.shape, all_feats.dtype))

    # gather labeled and unlabeled data separately
    inds1 = np.where(mask_lab == 1)[0]
    inds2 = np.where(mask_lab == 0)[0]
    num_labeled = len(inds1)

    labeled_feats = all_feats[inds1].detach().cpu().numpy()
    unlabeled_feats = all_feats[inds2]
    print('    labeled feats shape= {}, unlabeled feats shape= {}'.format(labeled_feats.shape, unlabeled_feats.shape))

    # sub-class split for each labeled class, to obtain sub-class center feature
    if subclass_split_num > 0:
        print('    Computing old class centers with subclass_split_num ', subclass_split_num)
        subclass_centers, subclass_gt_target = subclass_split(labeled_feats, all_labels[inds1], subclass_num=subclass_split_num)  # note how many sub-classes to split
        subclass_centers = torch.from_numpy(subclass_centers).cuda()
    else:
        subclass_centers = all_feats[inds1]
        subclass_gt_target = all_labels[inds1]

    if unlabeled_sampling:  # sample a subset of unlabeled instances for association
        np.random.seed(epoch)
        sel_inds = np.random.choice(range(len(unlabeled_feats)), replace=False, size=(int(sample_ratio*len(unlabeled_feats)),))
        unlabeled_feats = unlabeled_feats[sel_inds]
        #subset_gt_label = np.concatenate((all_labels[inds1], all_labels[inds2][sel_inds]))
        sel_inds += len(labeled_feats)  # actual position in the whole list
        #print('labeled subset + selected unlabeled subset: total gt class number= ', len(np.unique(subset_gt_label)))

    hybrid_feats = torch.cat((subclass_centers, unlabeled_feats))

    # semi-association
    if associate_type in ['thresholding', 'semi_dbscan']:
        new_all_labels = -1 * np.ones((len(hybrid_feats),), dtype=np.int64)
        new_all_labels[0:len(subclass_centers)] = subclass_gt_target  # only use labeled part

        t1 = time.time()
        print('    computing distance matrix for hybrid feature of shape {} and dtype {}...'.format(hybrid_feats.shape, hybrid_feats.dtype))
        if rerank:
            W = compute_jaccard_distance(hybrid_feats, k1=k1, k2=k2)
        else:
            W = 2 - 2 * torch.mm(hybrid_feats, hybrid_feats.t())  # 2-2X*Y
            W[W < 0] = 0
            W = W.sqrt().cpu().detach().numpy()
        
        W[0:len(subclass_centers), 0:len(subclass_centers)] = 100  # mask labeled data
        print('    distance matrix shape= {}, compute distance time usage= {} s'.format(W.shape, time.time() - t1))

        if associate_type == 'thresholding':
            print('    association using prior-constrained semi-association with thresh= ', thresh)
            temp_pseudo_label = association_algo(W, new_all_labels, len(subclass_centers), thresh)

        elif associate_type == 'semi_dbscan':
            print('    association using prior-improved Semi_DBSCAN with thresh= ', thresh)
            np.fill_diagonal(W, 0)
            semi_dbscan_algo = SemiDBSCAN(eps=thresh, min_samples=4, n_jobs=-1)
            temp_pseudo_label = semi_dbscan_algo.fit_predict(W, new_all_labels, all_instance_dist=(subclass_split_num<=0))

        del W

    elif associate_type == 'kmeans':
        assert (total_class_num > 0)
        t1 = time.time()
        print('    computing K-Means ({} classes) for hybrid feature of shape {}...'.format(total_class_num, hybrid_feats.shape))
        kmeans_algo = KMeans(n_clusters=total_class_num, random_state=10, n_init='auto')
        km = kmeans_algo.fit(hybrid_feats.cpu().numpy())
        temp_pseudo_label = km.labels_
        print('    K-means clustering time usage= {} s'.format(time.time()-t1))

    elif associate_type == 'ss_kmeans':
        t1 = time.time()
        print('    computing SS_KMeans ({} classes) for hybrid feature of shape {}...'.format(total_class_num, hybrid_feats.shape))
        kmeans = SemiSupKMeans(k=total_class_num, tolerance=1e-4, max_iterations=200, init='k-means++',
                           n_init=100, random_state=None, n_jobs=None, pairwise_batch_size=512, mode=None)  # default: max_iterations=10, n_init=10
        kmeans.fit_mix(unlabeled_feats, subclass_centers, torch.from_numpy(subclass_gt_target))
        temp_pseudo_label = kmeans.labels_.cpu().numpy()
        print('    K-means clustering time usage= {} s'.format(time.time()-t1))

    # convert hybrid pseudo label to per-instance pseudo label
    if subclass_split_num > 0:
        pseudo_label = -1*np.ones(all_labels.shape, dtype=all_labels.dtype)
        for lbl in np.unique(all_labels[inds1]):  #subclass_gt_target):
            img_inds = np.where(all_labels[inds1]==lbl)[0]
            subclass_ind = np.where(subclass_gt_target==lbl)[0]
            pseudo_label[img_inds] = temp_pseudo_label[subclass_ind[0]]

        if unlabeled_sampling:
            pseudo_label[sel_inds] = temp_pseudo_label[len(subclass_centers):]
        else:
            pseudo_label[num_labeled:] = temp_pseudo_label[len(subclass_centers):]

        del subclass_centers

    elif unlabeled_sampling:
            pseudo_label = -1*np.ones(all_labels.shape, dtype=all_labels.dtype)
            pseudo_label[0: num_labeled] = temp_pseudo_label[0:num_labeled]
            pseudo_label[sel_inds] = temp_pseudo_label[len(subclass_centers):]
    else:
        pseudo_label = temp_pseudo_label  # when association on all instances, labeled+unlabeled

    # =============================================
    # test pseudo label accuracy
    val_un_inds = np.where(pseudo_label[num_labeled:]>=0)[0]
    unlabel_train_acc = split_cluster_acc_v2(all_labels[inds2][val_un_inds], pseudo_label[num_labeled:][val_un_inds], old_or_new_mask[inds2][val_un_inds])
    unlabel_train_acc = [round(acc*100, 4) for acc in unlabel_train_acc]
    print('Test acc using backbone feat, on valid unlabeled train ({}/{} instances): Acc= {}'.format(len(val_un_inds), len(inds2), unlabel_train_acc))

    new_all_feats = torch.zeros(all_feats.shape, dtype=all_feats.dtype)
    new_all_feats[0:num_labeled] = all_feats[inds1]  # labeled 
    new_all_feats[num_labeled:] = all_feats[inds2]   # unlabeled 

    new_all_img_idxs = np.zeros(all_img_idxs.shape, dtype=np.int64)
    new_all_img_idxs[0:num_labeled] = all_img_idxs[inds1]
    new_all_img_idxs[num_labeled:] = all_img_idxs[inds2]

    # ======================= proxy memory related =============================
    # cluster-level proxy memory:
    proxy_memory, proxy_label, img_proxy_index, cluster_img_dict, all_cluster_mask = initialize_proxy_memory(
        pseudo_label, new_all_feats, new_all_img_idxs, num_labeled, outlier_thresh=outlier_thresh)

    ori_proxy_memory, ori_proxy_label = proxy_memory, proxy_label

    # assign the outliers to their nearest proxy class
    ot_inds = np.where(pseudo_label==-1)[0]
    if assign_outliers and len(ot_inds)>0:
        print('Assign outliers to nearest proxy class...')
        extended_pseudo_label = pseudo_label.copy()
        pseudo_label = torch.from_numpy(pseudo_label).long()
        ot_inds = []
        for lbl in torch.unique(pseudo_label):
            inds = torch.nonzero(pseudo_label==lbl).squeeze(-1)
            sup_inds = inds[inds < num_labeled]
            if (lbl==-1) or (len(sup_inds)==0 and len(inds)<=outlier_thresh):
                ot_inds.append(inds)

        ot_inds = torch.cat(ot_inds)
        ot_to_proxy_sim = torch.mm(new_all_feats[ot_inds].cuda(), ori_proxy_memory.detach().t())  # feature-to-proxy cosine similarity
        ot_assign_result = torch.argmax(ot_to_proxy_sim, dim=1)
        extended_pseudo_label[ot_inds] = ori_proxy_label[ot_assign_result].cpu().numpy()
        unlabel_train_acc = split_cluster_acc_v2(all_labels[inds2], extended_pseudo_label[num_labeled:], old_or_new_mask[inds2])
        unlabel_train_acc = [round(acc*100, 4) for acc in unlabel_train_acc]
        print('Test acc using associated anchor based assign, on unlabeled train ({} instances): Acc= {}'.format(len(inds2), unlabel_train_acc))

    del new_all_feats
    del new_all_img_idxs
    del all_feats
    del all_img_idxs
        
    return proxy_memory, proxy_label, img_proxy_index, cluster_img_dict, all_cluster_mask


"""
Re-initialize proxy memory and related proxy_label, img_proxy_index, etc...
  # pseudo_label: numpy array, indicating each instance's predicted label;
  # new_all_feats: each instance's feature;
  # new_all_img_idxs: each instance's unique image index
"""
def initialize_proxy_memory(pseudo_label, new_all_feats, new_all_img_idxs, num_labeled, outlier_thresh=10):
    pseudo_label = torch.from_numpy(pseudo_label).long()
    uniq_labels = torch.unique(pseudo_label)
    new_all_img_idxs = torch.from_numpy(new_all_img_idxs)
    print('new_all_img_idxs: min={}, max= {}, len= {}'.format(
        new_all_img_idxs.min(), new_all_img_idxs.max(), len(new_all_img_idxs)))

    cluster_img_dict = {}
    proxy_memory, proxy_label, per_cluster_size, all_cluster_mask = [], [], [], []
    img_proxy_index = -1 * torch.ones(new_all_img_idxs.shape).long()  # from unique image index to proxy index

    cnt = 0
    for lbl in uniq_labels:
        if lbl < 0: continue
        inds = torch.nonzero(pseudo_label == lbl).squeeze(-1)  # images whose pseudo label == lbl
        # do not consider new cluster whose size is smaller than outlier thresh
        sup_inds = inds[inds < num_labeled]
        if len(sup_inds) == 0 and len(inds) <= outlier_thresh: continue
        
        cluster_img_dict[int(lbl)] = new_all_img_idxs[inds]  # actual image indexes
        group_mask = len(inds)  # record cluster size as the mask

        proxy_feat = new_all_feats[inds].mean(0)
        proxy_memory.append(proxy_feat)
        proxy_label.append(lbl)
        img_proxy_index[new_all_img_idxs[inds]] = cnt
        per_cluster_size.append(len(inds))
        all_cluster_mask.append(group_mask)
        cnt += 1
    
    proxy_memory = torch.vstack(proxy_memory)
    proxy_memory = F.normalize(proxy_memory.detach(), dim=1).detach().cuda()
    proxy_label = torch.tensor(proxy_label).long().cuda()  # cluster label of each proxy in proxy_memory
    img_proxy_index = img_proxy_index.cuda()
    all_cluster_mask = torch.tensor(all_cluster_mask).float().cuda()

    print('After association: {} proxies out of {} clusters are generated, per-cluster size= {}'
          .format(len(proxy_memory), len(per_cluster_size), per_cluster_size))

    return proxy_memory, proxy_label, img_proxy_index, cluster_img_dict, all_cluster_mask



# Semi-supervised data association; The goal is:
# 1) to associate similar instances pairs whose distance<thresh,
# 2) make labeled data's association follow their ground truth prior,
# 3) avoid false associations such as A->b->C, where A and C are known to belong to different classes.
def association_algo(W, class_labels, num_labeled, thresh):
    t2 = time.time()

    # mask out the lower half
    for i in range(len(W)):
        lower_ind = np.arange(0, i + 1)
        W[i, lower_ind] = 1000

    print('Choosing candidates based on thresh {}'.format(thresh))
    ind = np.where(W < thresh)
    sorted_ind = np.argsort(W[ind[0], ind[1]])
    row_ind = ind[0][sorted_ind]  # from most similar to less similar
    col_ind = ind[1][sorted_ind]

    pseudo_labels = class_labels.copy()
    group_label_list = {}  # to record each group's current classes

    # first, initiate group_label_list by recording the labeled instances
    old_labels = class_labels[0:num_labeled]
    print('    labeled instances: unique label num= {}, min= {}, max= {}'.format(len(np.unique(old_labels)), old_labels.min(), old_labels.max()))
    for i in range(num_labeled):
        if class_labels[i] not in group_label_list:
            group_label_list[class_labels[i]] = [i]
        else:
            group_label_list[class_labels[i]].append(i)

    # then associate (labeled-to-unlabeled), or (unlabeled-to-unlabeled)
    pre_max_label = int(class_labels.max())  # starting number
    cnt = pre_max_label

    # during the following, pseudo_label and group_label_list are updated:
    for m in range(len(row_ind)):
        i = row_ind[m]
        j = col_ind[m]
        label1 = pseudo_labels[i]
        label2 = pseudo_labels[j]

        if label1==-1 and label2==-1:
            cnt += 1
            pseudo_labels[i] = cnt
            pseudo_labels[j] = cnt
            group_label_list[cnt] = [i, j]

        elif label1!=-1 and label2==-1:  # add the unlabeled instance to labeled instance's cluster
            group_label_list[label1].append(j)
            pseudo_labels[j] = label1

        elif label1==-1 and label2!=-1:
            group_label_list[label2].append(i)
            pseudo_labels[i] = label2

        elif label1!=-1 and label2!=-1 and label1!=label2:
            if (label1>pre_max_label or label2>pre_max_label):  # merge the two cluster if they are not both from labeled classes
                if label1>label2:
                    pseudo_labels[group_label_list[label1]] = label2  # assign label1 cluster instances as label2
                    group_label_list[label2].extend(group_label_list[label1])
                    group_label_list.pop(label1)
                elif label1<label2:
                    pseudo_labels[group_label_list[label2]] = label1  # assign label1 to all instances in cluster-label2
                    group_label_list[label1].extend(group_label_list[label2])
                    group_label_list.pop(label2)

    uniq_label_num = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
    assert (uniq_label_num == len(group_label_list))
    num_outliers = len(np.where(pseudo_labels==-1)[0])
    print('    Association_algo initially generates {} clusters in total, {}/{} are outliers.'.format(uniq_label_num, num_outliers, len(pseudo_labels)))

    # re-arrange labels to be continuous. i.e [0,1,2,3,...]
    cnt = 0
    for lbl in np.unique(pseudo_labels):
        if lbl >= 0:
            pseudo_labels[np.where(pseudo_labels == lbl)] = cnt
            cnt += 1
    print('    association_algo time usage= {} s'.format(time.time()-t2))

    return pseudo_labels


# split each labeled class into multiple sub-classes and return sub-class centers
# assume labeled_feats and class_labels are numpy array
def subclass_split(labeled_feats, class_labels, subclass_num=2):
    kmeans_algo = KMeans(n_clusters=subclass_num, random_state=10, n_init='auto')
    uniq_labels = np.unique(class_labels)
    subclass_centers = []
    subclass_gt_target = []
    per_subclass_num = []

    for lbl in uniq_labels:
        inds = np.where(class_labels == lbl)[0]
        temp_feats = labeled_feats[inds]
        if subclass_num > 1:
            km = kmeans_algo.fit(temp_feats)
            subclass_label = km.labels_
            for subclass in np.unique(subclass_label):
                subclass_feat = temp_feats[subclass_label==subclass]
                subclass_centers.append(subclass_feat.mean(0))
                subclass_gt_target.append(lbl)
                per_subclass_num.append(len(subclass_feat))
        else:
            subclass_centers.append(temp_feats.mean(0))
            subclass_gt_target.append(lbl)
            per_subclass_num.append(len(temp_feats))
    subclass_centers = np.vstack(subclass_centers)
    subclass_centers = subclass_centers / np.linalg.norm(subclass_centers, axis=1, keepdims=True)  # l2 normalization
    subclass_gt_target = np.array(subclass_gt_target)
    print('    image number of {} splited sub-class (old labeled): min= {}, max= {}'.format(len(per_subclass_num), min(per_subclass_num), max(per_subclass_num)))
    return subclass_centers, subclass_gt_target
