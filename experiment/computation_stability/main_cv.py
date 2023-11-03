import sys, os
import numpy as np
from dataeval.valuation import DataValuation
from dataeval.util import get_dataset, set_seed, get_seed

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, "/home/lixiang/PCS_SV")

dataset = "gaussian"
print(dataset)
trnX, trnY, devX, devY, tstX, tstY = get_dataset(dataset, '../data/{}.pkl'.format(dataset), noise=False)

# Define a DataValuation instance
dv = DataValuation(trnX, trnY, devX, devY)
dir = "../result/cv/{}".format(dataset)
if not os.path.exists(dir):
    os.makedirs(dir)

methods = ['tmc-shapley']
rep = 50

# Dictionary to store all vals for each method across repetitions
all_vals = {method: [] for method in methods}

for _ in range(rep):
    for method in methods:
        model = 'LR'
        print(method)
        set_seed(get_seed()+1)
        method_dict = dv.estimate(model, method)

        if method == 'tmc-shapley':
            for submethod, vals in method_dict.items():
                print(submethod, vals[:10])
                # Initialize submethod key if it doesn't exist
                if submethod not in all_vals:
                    all_vals[submethod] = []

                all_vals[submethod].append(vals)
        else:
            vals = np.array(list(method_dict.values()))
            all_vals[method].append(vals)
            print(method, vals[:10])

# Save the results to .npy files
for key, vals_list in all_vals.items():
    if len(vals_list) == 0:
        print(f"WARNING: No data for {key}. Skipping.")  # Optional, if you want to print it to console
        continue

    # Stack the vals_list and save to .npy file
    all_vals_stacked = np.stack(vals_list, axis=0)
    file_path = dir + "/SV_{}_{}.npy".format(key, model)
    np.save(file_path, all_vals_stacked)
