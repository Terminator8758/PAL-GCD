import numpy as np
from sklearn.neighbors import NearestNeighbors


"""A variant of the unsupervised DBSCAN algorithm, to support semi-supervised association such as GCD.
"""
class SemiDBSCAN():
    def __init__(self, eps=0.5, min_samples=4, n_jobs=None):
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs

    def fit_predict(self, X, y=None, all_instance_dist=False):
        """
        :param X: {array-like, distance matrix} of shape (n_samples, n_samples)
        :param y: label of each sample in X, value -1 indicates unknown label, otherwise known GT label
        :return: predicted pseudo label of shape (n_samples,)
        """
        if all_instance_dist:
            print('    masking distance of labeled positive pairs...')
            labeled_classes = np.unique(y)
            for cls in labeled_classes:
                if cls == -1: continue
                inds = np.where(y == cls)[0]
                for i in inds:
                    X[i, inds] = 0.01  # mask intra-class positives in labeled subset

        neighbors_model = NearestNeighbors(radius=self.eps, metric='precomputed', n_jobs=self.n_jobs)
        neighbors_model.fit(X)
        # threshold based neighbors search, return array of objects, sorted by increasing dist, first neighbor is self
        neigh_dist, neigh_ind = neighbors_model.radius_neighbors(X, return_distance=True, sort_results=True)

        # A list of all core samples found
        n_neighbors = np.array([len(neighbors) for neighbors in neigh_ind])
        core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=np.uint8)  # value=0(not core sample) or 1(core sample)
        labels = dbscan_inner(core_samples, neigh_ind, neigh_dist, y)

        num_clusters = len(set(labels))-1 if -1 in labels else len(set(labels))
        print('Semi-DBSCAN generates {} labels, min= {}, max= {}'.format(num_clusters, labels.min(), labels.max()))

        return labels


def dbscan_inner(is_core, neighborhoods, neigh_dist, y):
    label_num = 0
    stack = []
    labels = -1*np.ones((len(neighborhoods),), dtype=np.int64)

    # re-arrange neighbors of each sample, so that neighbors=[self, top-k, ..., top3, top2, top1]
    # also, obtain samples visit order by their dist to top1 neighbor
    min_neigh_dist = []
    for i in range(len(neighborhoods)):
        neighborhoods[i][1:] = neighborhoods[i][1:][::-1]
        dist = neigh_dist[i][1] if len(neigh_dist[i])>1 else 100
        min_neigh_dist.append(dist)

    visit_order = np.argsort(np.array(min_neigh_dist))  # distance increasing
    # visit_order = range(labels.shape[0])

    for i in visit_order:
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points. This is very similar to the classic
        # algorithm for computing connected components, the difference being that we label non-core points as
        # part of a cluster (component), but don't expand their neighborhoods.
        known_lbl = -1    # old class label of the group to be associated with i
        while True:
            if labels[i] == -1:
                labels[i] = label_num
                if y[i] >= 0:
                    known_lbl = y[i]  # assign the first-encountered known label to this group
                if is_core[i]:
                    neighb = neighborhoods[i]  # expand neighbors of i
                    for i in range(neighb.shape[0]):
                        v = neighb[i]
                        if labels[v] == -1 and (y[v]==known_lbl or y[v]==-1):
                            stack.append(v)

            if len(stack) == 0:
                break
            i = stack.pop()  # in while loop, the last one to pop is i itself

        label_num += 1

    #print('in dbscan_inner(): the first 100 proxies is_core= ', is_core[0:100])
    cnt = labels.max() + 1
    for lbl in np.unique(labels):
        inds = np.where(labels==lbl)[0]
        known_lbl = np.unique(y[inds])
        known_lbl = known_lbl[known_lbl != -1]
        if len(known_lbl) >= 2:
            print('associated cluster {}: known labels= {}, split into different clusters'.format(lbl, known_lbl))
            for j in known_lbl:
                if j >= 0:
                    labels[y==j] = cnt
                    cnt += 1

    print('{} known classes are assigned {} pseudo labels'.format(len(np.unique(y[y!=-1])), len(np.unique(labels[y!=-1]))))

    return labels



















