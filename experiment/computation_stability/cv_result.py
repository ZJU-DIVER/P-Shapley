import sys, os
import numpy as np

def compute_average_CV(vals_list):
    """
    Compute the average coefficient of variation (CV) for the given vals_list.

    vals_list: List of numpy arrays, each array contains the estimated Shapley values
               for each repetition of the method.
    """
    stacked_vals = np.stack(vals_list, axis=0)
    print(stacked_vals.shape)
    std_vals = np.std(stacked_vals, axis=0)
    mean_vals = np.mean(stacked_vals, axis=0)
    cv_vals = std_vals / mean_vals
    average_cv = np.mean(cv_vals)

    return average_cv

dataset = "gaussian"
dir = "../result/cv/{}".format(dataset)
methods = ['tmc_psv_square', 'tmc_psv_swish', 'tmc_psv_mish', 'tmc_psv', 'tmc-shapley']
model = 'LR'

with open(dir + "/result.txt", "w") as f:
    for method in methods:
        file_path = dir + "/SV_{}_{}.npy".format(method, model)
        if os.path.exists(file_path):
            loaded_vals_stacked = np.load(file_path)
            loaded_vals_list = [loaded_vals_stacked[i] for i in range(loaded_vals_stacked.shape[0])]
            average_CV = compute_average_CV(loaded_vals_list)
            result_str = f"Average CV for {method}: {average_CV:.5f}\n"
            f.write(result_str)
            print(result_str)  # Optional, if you want to print it to console as well
        else:
            print(f"WARNING: File for {method} not found. Skipping.")
