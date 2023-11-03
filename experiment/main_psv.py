import sys, os
# Import the evaluation task: Data Removal
from dataeval.valuation import DataValuation
from dataeval.eval import data_removal
from dataeval.metrics import weighted_acc_drop
import numpy as np
from dataeval.util import get_dataset
def test_concurrent(clf, method ):
    # This method test the concurrent evaluation of the data removal task
    return [1]*10

datasets = ["covertype", "wind", "fmnist_binary", "cifar-10_binary"]

for dataset in datasets:
    print(dataset)
    trnX, trnY, devX, devY, tstX, tstY = get_dataset(dataset, '../data/{}.pkl'.format(dataset), noise=False)
    # Define a DataValuation instance
    dv = DataValuation(trnX, trnY, devX, devY)
    dir = "../result/{}".format(dataset)
    if not os.path.exists(dir):
        os.makedirs(dir)
    methods = ['tmc-shapley']
    for method in methods:
        model = 'LR'
        print(method)
        method_dict = dv.estimate(model, method)
        if method == 'tmc-shapley':
            for submethod, vals in method_dict.items():
                vals_dict = {}
                for i, val in enumerate(vals):
                    vals_dict[i] = val
                accs = data_removal(vals_dict, trnX, trnY, tstX, tstY)
                print(vals[:10])
                np.save(dir + "/SV_{}_{}.npy".format(submethod, model), vals)
                with open(dir + "/ACC_{}_{}.csv".format(submethod, model), "w", encoding='utf-8') as f:
                    f.writelines(", ".join(["%.5f" % w for w in accs]))
                res = weighted_acc_drop(accs)
                print("The weighted accuracy drop of {} is {:.3f}".format(submethod, res))
        else:
            vals = np.array(list(method_dict.values()))
            np.save(dir + "/SV_{}_{}.npy".format(method, model), vals)
            accs = data_removal(method_dict, trnX, trnY, tstX, tstY)
            with open(dir + "/ACC_{}_{}.csv".format(method, model), "w", encoding='utf-8') as f:
                f.writelines(", ".join(["%.5f" % w for w in accs]))
            res = weighted_acc_drop(accs)
            print("The weighted accuracy drop of {} is {:.3f}".format(method, res))
