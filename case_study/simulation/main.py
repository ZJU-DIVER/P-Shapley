"""This script is used for check the proba of the event where acc is diff but proba is same."""

'''
In detail, we check the following events for clarity.

X: same acc
Y: same proba

Pr(X)
Pr(Y)

Pr(X|Y)
Pr(Y|X)
'''
import numpy as np


datasets = ['covertype', 'wind', 'fmnist', 'cifar10']
num_perm = 2500

def main():
    for dataset in datasets:
        data = load_data()
        n = len(y_train)
        # set counter
        num_same_acc, num_same_proba, num_sa_cond_sp, num_sp_cond_sa = 0
        
        old_au = init_score_acc()
        old_pu = init_score_proba()
        for _ in range(2500):
            perm = np.arange(n)
            np.random.shuffle(perm)
            for i in range(1, n+1):
                # get utility and then mariginal contirbution
                au = get_utility_acc()
                pu = get_utility_proba()
                if old_au == au:
                    num_same_acc += 1
                    if old_pu == pu:
                        num_sp_cond_sa += 1
                if old_pu == pu:
                    num_same_proba += 1
                    if old_au == au:
                        num_sa_cond_sp += 1
                        
                old_au = au
                old_pu = pu
                        
        print(f'num_same_acc: {num_same_acc}')
        print(f'num_same_proba: {num_same_proba}')
        print(f'num_sa_cond_sp: {num_sa_cond_sp}')
        print(f'num_sp_cond_sa: {num_sp_cond_sa}')