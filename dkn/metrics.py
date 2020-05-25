# Calculate AUC, MRR, NDCG@5, NDCG@10

import numpy as np
from sklearn.metrics import roc_auc_score

# refer to https://github.com/microsoft/recommenders/blob/master/reco_utils/recommender/deeprec/deeprec_utils.py
def group_labels(labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.

        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.

        Returns:
            all_labels: labels after group.
            all_preds: preds after group.

        """

        all_keys = list(set(group_keys))
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for l, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(l)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_labels, all_preds

def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    
    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def auc_score(labels, preds):
    auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
    return auc

def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def evaluate(labels, preds, group_keys):
    all_labels, all_preds = group_labels(labels, preds, group_keys)
    res = {}
    mean_mrr = np.mean(
        [   
            mrr_score(each_labels, each_preds)
            for each_labels, each_preds in zip(all_labels, all_preds)
        ]
    )
    res["mean_mrr"] = round(mean_mrr, 4)

    group_auc = np.mean(
        [
            roc_auc_score(each_labels, each_preds)
            for each_labels, each_preds in zip(all_labels, all_preds)
        ]
    )
    res["group_auc"] = round(group_auc, 4)

    for k in [5, 10]:
        ndcg_temp = np.mean(
            [   
                ndcg_score(each_labels, each_preds, k)
                for each_labels, each_preds in zip(all_labels, all_preds)
            ]
        )
        res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
    return round(group_auc, 4), round(mean_mrr, 4), res["ndcg@5"], res["ndcg@10"]