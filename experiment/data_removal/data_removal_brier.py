
from dataeval.eval import data_removal
from dataeval.options import args_parser
from dataeval.metrics import weighted_acc_drop, pr_curve
import pickle
import numpy as np
from dataeval.util import get_dataset

datasets = ["covertype", "wind", "fmnist_binary", "cifar-10_binary"]
# datasets = ["diabetes"]

# Define a DataValuation instance'
methods = ['tmc_psv_square', 'tmc_psv_swish', 'tmc_psv_mish', 'tmc_psv', 'tmc-shapley']
for dataset in datasets:
    data = pickle.load(open('../data/{}.pkl'.format(dataset), 'rb'))
    # The set of examples that will be evaluated
    trnX, trnY, devX, devY, tstX, tstY = get_dataset(dataset, '../data/{}.pkl'.format(dataset), noise=True)
    for method in methods:
        print(method)
        vals = np.load("../result/noise/{}/SV_{}.npy".format(dataset, method))
        remove_high_value=True
        # Sorting the indices of vals based on their values
        sorted_indices = list(np.argsort(vals))

        # Calling the pr_curve function with target_list and sorted_indices
        res = pr_curve(target_list, sorted_indices)
        print(res[2])
        # print("The weighted accuracy drop is {:.3f}".format(res))
