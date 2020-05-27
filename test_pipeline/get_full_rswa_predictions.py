#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:30:59 2020

@author: danielyaeger
"""
import pickle
from pathlib import Path
import numpy as np

def generate_full_rswa_predictions(name_of_staging: str,
                     path_to_staging: str,
                     name_of_apnea: str,
                     path_to_apnea: str,
                     name_of_predictions: str,
                     path_to_predictions: str,
                     save_name: str,
                     save_path: str,
                     sampling_rate: int = 10, 
                     epoch_length: int = 30):
    """Creates a dictionary keyed by ID with full-study rswa predictions for 
    each ID in the predictions file with name name_of_predictions in path
    path_to_predictions with name save_name in path save_path.
    """
    # Set up paths and load data
    if not isinstance(path_to_staging, Path):
        path_to_staging = Path(path_to_staging)
    
    with path_to_staging.joinpath(name_of_staging).open('rb') as fs:
        stage_dict = pickle.load(fs)
    
    if not isinstance(path_to_apnea, Path):
        path_to_apnea = Path(path_to_apnea)
    
    with path_to_apnea.joinpath(name_of_apnea).open('rb') as fa:
        apnea_dict = pickle.load(fa)
    
    if not isinstance(path_to_predictions, Path):
        path_to_predictions = Path(path_to_predictions)
    
    with path_to_predictions.joinpath(name_of_predictions).open('rb') as fp:
        predictions = pickle.load(fp)
    
    # create increment level to hold number of samples per epoch
    increment = sampling_rate*epoch_length

    # combine subsequences into one sequence for each sleeper
    sequence_dict = {}
    sleeper_IDs = list(set([ID.split('_')[0] for ID in list(predictions.keys())]))
    for ID in sleeper_IDs:
      #print(f'ID: {ID}')
      subseq = -1
      while subseq < 1e3:
        subseq += 1
        try:
          if ID not in sequence_dict:
            sequence_dict[ID] = predictions[f'{ID}_{subseq}']
          else:
            sequence_dict[ID] = np.concatenate((sequence_dict[ID], predictions[f'{ID}_{subseq}']))
        except:
          continue
    
    print('Generating full sequences...')
    rswa_full_predictions = {}
    for ID in sleeper_IDs:

        max_epoch = max(list(stage_dict[ID].keys()))
        rswa_full_predictions[ID] = np.zeros(max_epoch*increment)
        counter = 0
        for epoch in sorted(list(stage_dict[ID].keys())):
            if stage_dict[ID][epoch] in ['R','1','2','3']:
                if apnea_dict[ID][epoch] == 'None':
                    rswa_full_predictions[ID][(epoch-1)*increment:epoch*increment] = sequence_dict[ID][counter:counter + increment].argmax(-1)
                    counter += increment
                    if counter >= len(sequence_dict[ID]): break
    
    if not isinstance(save_path, Path): save_path = Path(save_path)
    
    with save_path.joinpath(save_name).open('wb') as fout:
        pickle.dump(rswa_full_predictions, fout)
    