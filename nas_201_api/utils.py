import pickle
import csv
import sys
import os
import numpy as np
from pprint import pprint


def read_in_preprocessed_data(path):
    """ Reading in the preprocessed data which is serialized as a dictionary"""
    with open (path, 'rb') as rf:
        record = pickle.load(rf)
    return record

def get_index_of_max_or_min_element_in_list(li, criterion=max):
    # note: we always get the first index if same max/min value occurs several times
    # note: we always get the first index if same max/min value occurs several times
    index = criterion(range(len(li)), key=li.__getitem__)
    return index

def get_index_and_element_of_max_or_min_element_in_list(li, criterion=max):
    index = criterion(range(len(li)), key=li.__getitem__)
    return index, li[index]

def write_dict_of_lists_to_csv(outfn, dict_to_store, include_keys_as_header_col = True, debug_mode=False):
    with open(outfn, 'w') as csvfile:
            if debug_mode:
                w = csv.writer(sys.stderr)
            else:
                w = csv.writer(csvfile)
            if include_keys_as_header_col:
                w.writerow(dict_to_store.keys())
            w.writerows(dict_to_store.values())
    if not debug_mode:
        print(f'Finished writing dict to {outfn}.')


def generate_summary_stats_from_list(li):
    a = np.asarray(li)
    summary = dict()
    summary['max'] = max(a)
    summary['min'] = min(a)
    summary['mean'] = np.mean(a)
    summary['median']= np.median(a)
    summary['std'] = np.std(a)
    summary['n_samples'] = len(li)
    return summary



"""
# NOTE: THIS IS PROBABLY NOT THE ONLY RS BASELINE WE WANT..
# WE WANT RATHER TO COMPARE FOR EVERY RANDOM SEED which result we obtain for all candidate algos (incl. a ranking)
# still useful to test for significance
# TODO: decide 

def generate_random_search_baseline(random_seed_dict, precomputed_acc_dict, budgets, max_number_of_epochs=200):
    # here we generate the RS baseline for a all search algorithms
    # we assume a predefined number of budgets, for which we analogously simulate each search algorithm
    # we assume: one run of RS = one full function evaluation = evaluating one arch for the max of epochs available (here 200)
    # we always take the first n sampled archs in every entry of random_seed_dict
    # as such we have an overlap between the simulation and the RS baseline
    rs_baselines = dict()
    for b in budgets:
        results_across_random_seeds = list()
        assert b >= max_number_of_epochs, "The budget has to be large enough to allow for one full function evaluation."
        n_possible_RS_runs = int(b/max_number_of_epochs)
        for i in random_seed_dict:
            relevant_indices = random_seed_dict[i][:n_possible_RS_runs]
            best_result = max([max(precomputed_acc_dict[index]) for index in relevant_indices])
            results_across_random_seeds.append(best_result)
            summary_stats = generate_summary_stats_from_list(results_across_random_seeds)
            rs_baselines[b] = summary_stats
    return rs_baselines


budgets = [200, 1000, 1000,5000,10000,50000,100000,500000,1000000]
rs_seed_dict = read_in_preprocessed_data('/Users/d071503/Data/Data_NASBENCH201/out/precomputed_random_samples.pkl')
acc_dict = read_in_preprocessed_data('/Users/d071503/Data/Data_NASBENCH201/out/proc_data_cifar10-valid.pkl')
rs_baseline = generate_random_search_baseline(rs_seed_dict,acc_dict, budgets, max_number_of_epochs=200)
pprint(rs_baseline)



somedict = {'raymond':['red','r'], 'rachel':['blue','b'], 'matthew':['green','g']}
write_dict__of_lists_to_csv('./test', somedict, debug_mode=True)

somedict = {'raymond':['red','r'], 'rachel':['blue','b'], 'matthew':['green','g']}
write_dict_of_lists_to_csv('./test.csv', somedict, include_keys_as_header_col=False, debug_mode=False)


l = [1,2,10,4,10,10]
i = get_index_of_max_or_min_element_in_list(l)
print(i)

"""
