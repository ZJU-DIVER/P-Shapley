import sys, os
# Import the evaluation task: Data Removal
from dataeval.valuation import DataValuation
from dataeval.eval import data_removal
from dataeval.metrics import pr_curve
from sklearn import metrics
import numpy as np
from dataeval.util import get_dataset

datasets = ["covertype", "wind", "fmnist_binary", "cifar-10_binary"]
model = 'LR'
result = dict()
for dataset in datasets:
    trnX, trnY, devX, devY, tstX, tstY = get_dataset(dataset, '../data/{}.pkl'.format(dataset), noise=True)
    methods = ['loo', 'beta-shapley', 'tmc-shapley']
    for method in methods:
        print(datasets)
        print(method)
        # Define a DataValuation instance
        dv = DataValuation(trnX, trnY, devX, devY)
        method_dict = dv.estimate(model, method)

        dir = "../result/noise/{}".format(dataset)
        if not os.path.exists(dir):
            os.makedirs(dir)

        if method == 'tmc-shapley':
            for submethod, vals in method_dict.items():
                np.save(dir + "/SV_{}_{}.npy".format(submethod,model), vals)
                target_list = [0 if (i % 10 == 1) else 1 for i in range(len(vals))]
                # Calling the pr_curve function with target_list and sorted_indices
                res = metrics.roc_auc_score(target_list, vals)
                print(res)
                if submethod not in result:
                    result[submethod] = []
                result[submethod].append(res)
        else:
            vals = np.array(list(method_dict.values()))
            np.save(dir + "/SV_{}_{}.npy".format(method,model), vals)
            target_list = [0 if (i % 10 == 1) else 1 for i in range(len(vals))]
            res = metrics.roc_auc_score(target_list, vals)
            if method not in result:
                result[method] = []
            result[method].append(res)
            print(res)

with open("../result/noise/AUC_{}.txt".format(model), 'w') as file:
    for key, values in result.items():
        values_str = " & ".join("{:.3f}".format(value) for value in values)  # Format each value to have 3 decimal places
        file.write(f"{key} & {values_str}\n")
