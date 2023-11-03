
import numpy as np

from sklearn.metrics import auc

def weighted_acc_drop(accs, metric='acc'):
    ''' Weighted accuracy drop, please refer to (Schoch et al., 2022)
        for definition
    '''
    # accs = copy.copy(accs)
    accs.append(0.)
    accs = np.array(accs)
    diff = accs[:-1] - accs[1:]
    c_sum = np.cumsum(diff)
    weights = np.array(list(range(1, diff.shape[0]+1)))
    weights = 1.0/weights
    score = weights * c_sum
    return score.sum()

def pr_curve(target_list, ranked_list):
    ''' Compute P/R for two given lists and plot the P/R curve
    '''
    p, r = [], []

    # Iterating from the third element to the end of ranked_list
    for idx in range(3, len(ranked_list)+1):
        partial_list = ranked_list[:idx]

        # Calculating the intersection of target_list and partial_list
        union = list(set(target_list) & set(partial_list))

        # Calculating and appending recall to the r list
        r.append(float(len(union))/len(target_list))

        # Calculating and appending precision to the p list
        p.append(float(len(union))/len(partial_list))

    # Calculating the area under the Precision-Recall curve
    score = auc(r, p)
    return p, r, score

if __name__ == "__main__":
    acc = [0.8, 0.7, 0.6]
    print(weighted_acc_drop(acc))