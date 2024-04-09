'''
Author: LX
Date: 2024-04-09 13:50:08
LastEditTime: 2024-04-09 14:43:30
Description: 
'''
import numpy as np
from typing import Optional, Any
from tqdm import tqdm
from .util import get_utility, get_utility_prob, get_seed
import concurrent.futures



# def gelu(input_array):
#     cdf = 0.5 * (1.0 + erf(input_array / np.sqrt(2.0)))
#     return input_array * cdf

def truncated_mc_helper(args):
    seeds, cnt, x_train, y_train, x_valid, y_valid, clf, T, epsilon = args
    N = len(y_train)
    num_classes = len(list(set(y_train)))
    idxes = list(range(N))
    np.random.seed(seeds)
    np.random.shuffle(idxes)
    final_acc = get_utility(x_train, y_train, x_valid, y_valid, clf)
    
    acc = np.zeros(cnt)
    new_acc = np.zeros(cnt)
    val = np.zeros([cnt, N])
    for i in range(1, N + 1):
        if abs(final_acc - acc[0]) < epsilon:
            break
        x_temp, y_temp = x_train[idxes[:i], :], y_train[idxes[:i]]
        # cal tmc-shapley
        new_acc[0] = get_utility(x_temp, y_temp, x_valid, y_valid, clf)
        val[0, idxes[i - 1]] += new_acc[0] - acc[0]
        # cal tmc_psvW
        new_acc[1] = get_utility_prob(x_temp, y_temp, x_valid, y_valid, clf, num_classes)
        val[1, idxes[i - 1]] += new_acc[1] - acc[1]
        # cal tmc_psv_square
        new_acc[2] = new_acc[1] * new_acc[1]
        val[2, idxes[i - 1]] += new_acc[2] - acc[2]
        # cal swish: x * sigmoid(x)
        new_acc[3] = new_acc[1] / (1 + np.exp(- new_acc[1]))
        val[3, idxes[i - 1]] += new_acc[3] - acc[3]
        # cal mish
        new_acc[4] = new_acc[1] * np.tanh(np.log(1 + np.square(new_acc[1])))
        val[4, idxes[i - 1]] += new_acc[4] - acc[4]
        # AUC
        new_acc[5] = get_utility(x_temp, y_temp, x_valid, y_valid, clf, metric='auc')
        val[5, idxes[i - 1]] += new_acc[5] - acc[5]
        # Logloss
        new_acc[6] = get_utility(x_temp, y_temp, x_valid, y_valid, clf, metric='log-loss')
        val[6, idxes[i - 1]] += new_acc[6] - acc[6]
        # new_acc[6] = get_utility(x_temp, y_temp, x_valid, y_valid, clf, metric='f1-score')
        # val[6, idxes[i - 1]] += new_acc[6] - acc[6]
        # # cal sigmoid
        # new_acc[7] = 1 / (1 + np.exp(- new_acc[1]))
        # val[7, idxes[i - 1]] += new_acc[7] - acc[7]
        # # cal tanh
        # new_acc[8] = np.tanh(new_acc[1])
        # val[8, idxes[i - 1]] += new_acc[8] - acc[8]
        # update accs
        acc = new_acc.copy()
        # print(acc)
    return val

Array = np.ndarray
def truncated_mc(
        x_train: Array,
        y_train: Array,
        x_valid: Optional[Array] = None,
        y_valid: Optional[Array] = None,
        clf: Optional[Any] = None,
        num_perm: int = 2500,
        truncated_threshold: float = 0.001
) -> dict:
    N = len(y_train)
    T = num_perm
    epsilon = truncated_threshold
    np.random.seed(get_seed())
    seeds = np.random.randint(0, 100000, T)
    cnt = 10
    args = [(seeds[i], cnt, x_train, y_train, x_valid, y_valid, clf, T, epsilon) for i in range(T)]
    # Implement the num_perm loop with concurrent futures
    workers = 20
    vals = np.zeros([cnt, N])
    test = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(truncated_mc_helper, arg) for arg in args]
        for future in tqdm(concurrent.futures.as_completed(futures), total=T, desc="Calculating"):
            val = future.result()
            for i in range(len(val)):
                vals[i] += val[i]
    print(test)
    return {
        'tmc-shapley': vals[0] / T, 'tmc_psv': vals[1] / T, 'tmc_psv_square': vals[2] / T, \
            'tmc_psv_swish': vals[3] / T, 'tmc_psv_mish': vals[4] / T, 'tmc-shapley_auc': vals[5] / T, 'tmc-shapley_logloss': vals[6] / T
            }