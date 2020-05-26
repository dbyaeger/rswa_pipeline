#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:06:25 2020

@author: danielyaeger
"""

import pickle
from pathlib import Path
import numpy as np

def generate_targets(name_of_ground_truth_staging: str,
                     path_to_ground_truth_staging: str,
                     name_of_ground_truth_apnea: str,
                     path_to_ground_truth_apnea: str,
                     path_to_p_files: str,
                     save_path: str,
                     sampling_rate: int = 10, 
                     epoch_length: int = 30):
    """Creates a dictionary keyed by ID with full-study rswa targets for each
    ID with the name RSWA_ground_truth_full_targets.p' in the save_path. THE 
    name_of_ground_truth_staging AND name_of_ground_truth_apnea SHOULD
    CORRESPOND TO HUMAN LABELED DATA or targets WILL NOT BE VALID. THE 
    path_to_p_files SHOULD CORRESPOND to p_files using GROUND TRUTH STAGING
    or LABELS WILL BE INVALID.
    """
    
    # Set up paths
    if not isinstance(path_to_ground_truth_staging, Path):
        path_to_ground_truth_staging = Path(path_to_ground_truth_staging)
    
    if not isinstance(path_to_ground_truth_apnea, Path):
        path_to_ground_truth_apnea = Path(path_to_ground_truth_apnea)
    
    apnea_dict_path = path_to_ground_truth_apnea.joinpath(name_of_ground_truth_apnea)
    #print(apnea_dict_path)
    
    # Set up data generator
    truth_gen = DataGeneratorAllWindows(path_to_p_files, apnea_dict_path, 
                                       batch_size=1, window_size=10)           
    rem_targets_dict = {}
    
    # iterate over IDs and get targets FOR REM SLEEP ONLY
    for ID in truth_gen.list_IDs:        
        _, y = truth_gen.__getitem_for_ID__(ID)        
        rem_targets_dict[ID] = y.argmax(-1)
    
    # create full-length predictions
    with path_to_ground_truth_staging.joinpath(name_of_ground_truth_staging).open('rb') as fs:
        stage_dict = pickle.load(fs)
    
    with apnea_dict_path.open('rb') as fa:
        apnea_dict = pickle.load(fa)
    
    # create increment level to hold number of samples per epoch
    increment = sampling_rate*epoch_length

    # combine subsequences into one sequence for each sleeper
    sequence_dict = {}
    sleeper_IDs = list(set([ID.split('_')[0] for ID in truth_gen.list_IDs]))
    for ID in sleeper_IDs:
      print(f'ID: {ID}')
      subseq = -1
      while subseq < 1e3:
        subseq += 1
        try:
          if ID not in sequence_dict:
            sequence_dict[ID] = rem_targets_dict[f'{ID}_{subseq}']
          else:
            sequence_dict[ID] = np.concatenate((sequence_dict[ID], rem_targets_dict[f'{ID}_{subseq}']))
        except:
          continue
    
    print('Generating full sequences...')
    rswa_full_targets = {}
    for ID in sleeper_IDs:

        max_epoch = max(list(stage_dict[ID].keys()))
        rswa_full_targets[ID] = np.zeros(max_epoch*increment)
        counter = 0
        for epoch in sorted(list(stage_dict[ID].keys())):
            if stage_dict[ID][epoch] in ['R','1','2','3']:
                if apnea_dict[ID][epoch] == 'None':
                    rswa_full_targets[ID][(epoch-1)*increment:epoch*increment] = sequence_dict[ID][counter:counter + increment]
                    counter += increment
                    if counter >= len(sequence_dict[ID]): break
    
    if not isinstance(save_path, Path): save_path = Path(save_path)
    
    with save_path.joinpath('RSWA_ground_truth_full_targets.p').open('wb') as fout:
        pickle.dump(rswa_full_targets, fout)
