import pickle
import csv
import sys
import os


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




"""

somedict = {'raymond':['red','r'], 'rachel':['blue','b'], 'matthew':['green','g']}
write_dict__of_lists_to_csv('./test', somedict, debug_mode=True)

somedict = {'raymond':['red','r'], 'rachel':['blue','b'], 'matthew':['green','g']}
write_dict_of_lists_to_csv('./test.csv', somedict, include_keys_as_header_col=False, debug_mode=False)


l = [1,2,10,4,10,10]
i = get_index_of_max_or_min_element_in_list(l)
print(i)

"""
