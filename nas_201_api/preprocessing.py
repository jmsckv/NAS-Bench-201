import os
import numpy as np
import pickle
import random
from tqdm import tqdm
import logging
from nas_201_api import NASBench201API as API
from utils import get_index_of_max_or_min_element_in_list, write_dict_of_lists_to_csv, dict_to_array, rank_columns_of_2d_array


RAWPATH = os.environ['NASBENCH_RAW']
OUTPATH = os.environ['NASBENCH_OUT']
default_datasets = ['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120']
default_seeds = [777, 888, 999]

# TODO: rework, logging
def validate_data(api=None, datasets=default_datasets, seeds=default_seeds, rawpath=RAWPATH, outpath=OUTPATH, test_run=None):
    """
    Goal is to find out for which architectures we have 200 epochs on all datasets with 3 random seeds.
    This claim is made in the paper, but as of 23/2/20 the dataset contains less observations.
    :param datasets: datasets contained in NASBENCH201
    :param seeds: random seeds for which potentially every architecture was trained on potentially every dataset
    :return tuple
    """
    # in the test run case, we simply randomly draw a specified number of architectures
    if test_run:
        n_archs = random.sample(range(len(api)), test_run)
    else:
        n_archs = range(len(api))

    # validate the number of runs for every architecture in every dataset
    n_runs = dict()

    # which exceptions do we get for which architecture?
    exceptions = {key: set() for key in n_archs}

    # validation logic
    for d in tqdm(datasets):
        n_runs[d] = list()  # a list with each entry representing an architecture
        for i in n_archs:
            try:
                results_i = api.query_meta_info_by_index(i)
                count_runs = 0 # how many runs with different random seeds?
                for s in seeds:
                    try:
                        k = (d, s)
                        if results_i.all_results[k].epochs == 200:  # we assume that this attribute indicates a valid run, another criterion may be better
                            count_runs += 1
                    except Exception as e:
                        exceptions[i].update({(type(e),e)})
                n_runs[d].append(count_runs)
            except Exception as e:
                exceptions[i].update({(type(e),e)})

    # serialize results
    results = (n_runs, exceptions)
    if outpath:
        res_path = OUTPATH + '/validation_results.pkl'
        with open(res_path,'wb') as wf:
            pickle.dump(results,wf)

    return results

# validate data
#api = API(RAWPATH)
#api.get_more_info(2,'cifar10-valid',1)
#api.get_more_info(2,'cifar10-valid',100)
#r_runs, exceptions = validate_data(api)



def create_record(outpath,api, log_exceptions = True,dataset='cifar10-valid',epochs=200, best_so_far=True, store_CSV =True, store_np=True, precompute_ranks_per_epoch=True):
    """
    :param dataset: dataset for which we query api
    :param outpath: specify path where returned df gets serialized as CSV file
    :return: a dictionary , where each index/entry corresponds to one architecture, and each of the 1-200 columns corresponds to the validation accuracy of the corresponding period
    """

    if log_exceptions:
        log_path = os.path.join(outpath, 'create_record.log')
        logger = logging.getLogger()
        # Configure logger
        logging.basicConfig(filename=log_path, format='%(asctime)s %(filename)s: %(message)s', filemode='w')

    results = {key:list() for key in range(len(api))}
    # results_test = {key:None for key in range(len(api))} currently api does not support easy lookup of test

    exceptions = list()

    for i in range(len(api)):
        for e in range(epochs):
            acc = 0
            try:
                if best_so_far:
                    current_acc = api.get_more_info(i,dataset,e)['valid-accuracy']
                    if acc < current_acc:
                        acc = current_acc
                else:
                    acc =  api.get_more_info(i,dataset,e)['valid-accuracy']
                results[i].append(acc)
            except Exception as e:
                logging.error(type(e),e, exc_info=True)
                exceptions.append((i,e))
    if len(exceptions):
            print(f"Found {len(exceptions)} exceptions. Corresponds to a fraction of:  {len(exceptions)/(len(api)*epochs)}")

    """
    # currently disabled as api not reporting test performance via get_more_info
    # let's also record for each architecture how well it performs on the held-out test data
    # therefore we take the best performing weights on val and look up the corresponding performance on test
    for key in results.keys():
        ind = get_index_of_max_or_min_element_in_list(results[key])
        try:
            results_test[key] = api.get_more_info(key,'cifar10-valid', ind)
        except Exception as e:
            logging.error(type(e), e, exc_info=True)
    # would also have to serialize
    """

    # serialize data, this is what we'll use for simulation
    out_fn = os.path.join(outpath,'proc_data_'+ dataset +'.pkl')
    with open (out_fn, 'wb') as wf:
        pickle.dump(results,wf)
    if store_CSV:
        out_fn = os.path.join(outpath, 'proc_data_' + dataset + '.csv')
        write_dict_of_lists_to_csv(out_fn, results, include_keys_as_header_col=False)
    if store_np:
        out_fn = os.path.join(outpath,  'proc_data_' + dataset + '.npy')
        np.save(out_fn, dict_to_array(results))
    if precompute_ranks_per_epoch:
        out_fn = os.path.join(outpath, 'precomputed_ranks_per_epoch.npy')
        np.save(out_fn, rank_columns_of_2d_array(dict_to_array(results)))

    print(f"Finished preprocessing for dataset {dataset}.")

    results, exceptions


def shuffle_data_n_times_and_store(outpath, max_i, n_samples=500, store_CSV = True):
    """ This is to prevent that same random seed could return different random sequences on different hardware.
     Hence, we draw all random once and before running the experiments.
    The serialized results are also used to simulate RS baselines."""
    results = {}
    for i in range(n_samples):
        random.seed(i)
        l = list(range(max_i))
        random.shuffle(l)
        results[i] = l
    out_fn = os.path.join(outpath, 'precomputed_random_samples.pkl')
    with open (out_fn,'wb') as wf:
        pickle.dump(results,wf)
    if store_CSV:
        out_fn = os.path.join(outpath, 'precomputed_random_samples.csv')
        write_dict_of_lists_to_csv(out_fn, results, include_keys_as_header_col=False)



def main():
    api = API(RAWPATH)
    r,e = create_record(outpath=OUTPATH,api=api)
    shuffle_data_n_times_and_store(OUTPATH,len(api))

main()
