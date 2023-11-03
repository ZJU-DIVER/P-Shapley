import sys, os
from dataeval.eval import data_removal, data_removal_brier, data_removal_loss
from dataeval.options import args_parser
from dataeval.metrics import weighted_acc_drop, pr_curve
import pickle
import numpy as np
from dataeval.util import get_dataset

datasets = ["covertype", "wind", "fmnist_binary", "cifar-10_binary"]
# datasets = ["diabetes"]
model = "SVC"
# Define a DataValuation instance'
methods = ['loo', 'beta-shapley', 'tmc-shapley', 'tmc-shapley_auc', 'tmc_psv', 'tmc_psv_square', 'tmc_psv_swish', 'tmc_psv_mish']
for dataset in datasets:
    data = pickle.load(open('../data/{}.pkl'.format(dataset), 'rb'))
    # The set of examples that will be evaluated
    trnX, trnY, devX, devY, tstX, tstY = get_dataset(dataset, '../data/{}.pkl'.format(dataset))
    dir = "../result/{}".format(dataset)
    if not os.path.exists(dir):
        os.makedirs(dir)
    results = {method: [] for method in methods}
    for method in methods:
        print(method)
        vals = np.load(dir + "/SV_{}_{}.npy".format(method, model))
        vals_dict = {}
        for i, val in enumerate(vals):
            vals_dict[i] = val
        accs = data_removal(vals_dict, trnX, trnY, tstX, tstY)
        # with open(dir + "/ACC_{}_{}.csv".format(method, model), "w", encoding='utf-8') as f:
        #     f.writelines(", ".join(["%.5f" % w for w in accs]))
        results[method].append(weighted_acc_drop(accs))
        accs = data_removal_brier(vals_dict, trnX, trnY, tstX, tstY)
        results[method].append(-1 * weighted_acc_drop(accs))
        accs = data_removal_loss(vals_dict, trnX, trnY, tstX, tstY)
        results[method].append(-1 * weighted_acc_drop(accs))

    with open(dir + "/WAD_{}.txt".format(model), 'w') as file:
        for key, values in results.items():
            values_str = " & ".join(
                "{:.3f}".format(value) for value in values)  # Format each value to have 3 decimal places
            file.write(f"{key} & {values_str}\n")