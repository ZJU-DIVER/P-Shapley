import sys, os
import argparse
# Import the evaluation task: Data Removal
from dataeval.valuation import DataValuation
from dataeval.eval import data_removal
from dataeval.metrics import weighted_acc_drop
import numpy as np
from dataeval.util import get_dataset, get_seed
import random


def run(dataset, model):
    try:
        trnX, trnY, devX, devY, tstX, tstY = get_dataset(dataset, '../data/{}.pkl'.format(dataset), noise=False)

    except Exception as e:
        print(f"Error while loading dataset {dataset}: {e}")
        return  # Skip the rest of this iteration and proceed to the next dataset

    # Define a DataValuation instance
    dv = DataValuation(trnX, trnY, devX, devY)
    
    dir = "../upload_result/{}".format(dataset)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except Exception as e:
            print(f"Error while creating directory {dir}: {e}")
            return

    methods = ['tmc-shapley', 'loo', 'beta-shapley' ]
    for method in methods:
        print(method)
        
        try:
            method_dict = dv.estimate(model, method, dataset)
        except Exception as e:
            print(f"Error during estimation with method {method}: {e}")
            continue

        if method == 'tmc-shapley':
            for submethod, vals in method_dict.items():
                vals_dict = {}
                for i, val in enumerate(vals):
                    vals_dict[i] = val
                
                try:
                    accs = data_removal(vals_dict, trnX, trnY, tstX, tstY)
                except Exception as e:
                    print(f"Error during data removal for submethod {submethod}: {e}")
                    continue

                print(vals[:10])

                try:
                    np.save(dir + "/SV_{}_{}.npy".format(submethod, model), vals)
                    with open(dir + "/ACC_{}_{}.csv".format(submethod, model), "w", encoding='utf-8') as f:
                        f.writelines(", ".join(["%.5f" % w for w in accs]))
                except Exception as e:
                    print(f"Error while saving results for submethod {submethod}: {e}")

                # try:
                #     res = weighted_acc_drop(accs)
                #     print("The weighted accuracy drop of {} is {:.3f}".format(submethod, res))
                # except Exception as e:
                #     print(f"Error computing weighted accuracy drop for submethod {submethod}: {e}")

        else:
            try:
                vals = np.array(list(method_dict.values()))
                np.save(dir + "/SV_{}_{}.npy".format(method, model), vals)
                accs = data_removal(method_dict, trnX, trnY, tstX, tstY)
                with open(dir + "/ACC_{}_{}.csv".format(method, model), "w", encoding='utf-8') as f:
                    f.writelines(", ".join(["%.5f" % w for w in accs]))
                res = weighted_acc_drop(accs)
                print("The weighted accuracy drop of {} is {:.3f}".format(method, res))
            except Exception as e:
                print(f"Error during non-tmc-shapley method processing for method {method}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run data valuation methods.')
    parser.add_argument('--model', type=str, required=True, help='Model to use for evaluation.')
    parser.add_argument('--datasets', nargs='+', default=['covertype', 'wind', 'fmnist_binary', 'cifar-10_binary'], help='Datasets to evaluate.')
    args = parser.parse_args()
    # Set a fixed seed for reproducibility
    SEED = get_seed()
    np.random.seed(SEED)
    random.seed(SEED)
    for dataset in args.datasets:
        print(dataset)
        run(dataset, args.model)