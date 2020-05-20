#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:52:19 2020

@author: danielyaeger
"""

from pathlib import Path
import pickle

def delete_files_not_in_test_set(path_to_directory: str):
    """Deletes files that correspond to IDs that aren't in the test partition 
    of master_ID_list.p from a direcory. Assumes that master_ID_list.p is in 
    the directory. Also generates a file called data_partition.p that contains 
    all the files in the directory that are in the directory"""
    
    if not isinstance(path_to_directory, Path):
        path_to_directory = Path(path_to_directory)
    
    with path_to_directory.joinpath('master_ID_list.p').open('rb') as fh:
        test_IDs = list(pickle.load(fh)['test'])
    
    keep_files = set()
    for file in path_to_directory.iterdir():
        if file.name.startswith('X'):
            ID = file.name.split('_')[0]
            if ID in test_IDs:
                keep_files.add(file.name.split('.')[0])
            else:
                file.unlink()
    
    data_partition = {'test': keep_files}
    
    with path_to_directory.joinpath('data_partition.p').open('wb') as fout:
        pickle.dump(data_partition, fout)

        
    
    