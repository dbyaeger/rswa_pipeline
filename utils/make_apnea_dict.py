#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:08:57 2020

@author: danielyaeger
"""
import numpy as np
from pathlib import Path
import pickle
from data_generators.data_generator_apnea import DataGeneratorApnea

def make_ground_truth_apnea_dict(path_to_data: str,
                    save_path: str, apnea_threshold_for_epoch: int = 10,
                    sampling_rate: int = 10, epoch_length: int = 30):
    """Creates ground truth apnea labels at epoch level and stores these in
    a dictionary keyed by ID. The path_to_data MUST BE FOR apnea data
    created using human (ground truth) staging or the ground truth apnea
    labels WILL NOT BE accurate. Outputs a file named ground_truth_apnea_dict.p
    in the save_path with the format:
        
        ground_truth_apnea_dict = {ID_0: {1: 'None', 2: 'A/H', 3: 'None', ....},
        ...}
        
    where 'None' means no apnea/hypoxia for that epoch and 'A/H' means apnea/
    hypoxia for the epoch.
    """
    # Set up paths
    if not isinstance(path_to_data, Path):
        path_to_data = Path(path_to_data)
    
    with path_to_data.joinpath('stage_dict.p').open('rb') as fh:
        stage_dict = pickle.load(fh)
    
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    
    # Adjust apnea_threshold_for_epoch to be in units of samples
    apnea_threshold_for_epoch *= sampling_rate
    
    # Create increment to store number of samples per epoch
    increment = sampling_rate*epoch_length
    
    # Make data generator
    test_gen = DataGeneratorApnea(data_path = path_to_data, mode="test")         
    IDs = test_gen.IDs      
    signal_level_labels = {ID: {} for ID in IDs}
    
    # iterate over IDs and generate predictions
    for ID in IDs:
        test_gen = DataGeneratorApnea(n_classes = 2,
                                 data_path = path_to_data,
                                 single_ID = ID,
                                 batch_size = 16,
                                 mode="test",
                                 context_samples=300,
                                 shuffle = False,
                                 use_staging = True,
                                 REM_only = False)
        signal_level_labels[ID] = np.array(test_gen.labels)
    
    # Get epoch-level labels
    epoch_level_labels = {}
    for ID in IDs:
        epoch_level_labels[ID] = {epoch: 'None' for epoch in stage_dict[ID]}
        counter = 0
        for epoch in sorted(list(stage_dict[ID].keys())):
            if stage_dict[ID][epoch] in ['R','1','2','3']:
                if signal_level_labels[ID][counter:counter + increment].sum() >= apnea_threshold_for_epoch:
                    epoch_level_labels[ID][epoch] = 'A/H'
                counter += increment
                if counter >= len(signal_level_labels[ID]): break
    
    with save_path.joinpath('ground_truth_apnea_dict.p').open('wb') as fout:
        pickle.dump(epoch_level_labels, fout)

def make_apnea_dict(signal_level_predictions_name: str, predictions_path: str,
                    stage_file_name: str, stage_path: str, save_name: str,
                    save_path: str, apnea_threshold_for_epoch: int = 10,
                    sampling_rate: int = 10, epoch_length: int = 30) -> None:
    """
    Creates an apnea_dict for all IDs present in the signal_level_predictions_name
    file. The apnea dict is in the format:
        
        apnea_dict = {'predictions': {ID: [0: 'None', 1: 'None', 2: 'A/H', ...]},..},
                      'targets': {ID: [0: 'None', 1: 'A/H', 2: 'None', ...]},..}}
        
        where the key is an integer that indicates the epoch and the 
        corresponding value is either 'None' or 'A/H'.
    
    The apnea dict is saved in the save_path.
    INPUTS:
        signal_level_predictions_name: name of file containing predictions (only for sleep epochs),
            which is a pickled dictionary, keyed by ID
        predictions_path: path where signal_level_predictions_name file lives
        stage_file_name: name of staging file
        stage_path: where stage file lives
        save_name: name to give output file
        save_path: where to save output file
        apnea_threshold_for_epoch: the total length of apneic events (in s) 
            required for a sleep epoch to be called an apnea epoch
        sampling_rate: sampling_rate at which labels were assigned, typically 10 Hz
        epoch_length: epoch_length in seconds, typically 30 s
    
    RETURNS:
        dictionary, keyed by epoch, with a value of 'A/H' indicating apnea/hypopnea
        and 'None' indicating no apnea/hypopnea
    """
    if not isinstance(predictions_path, Path):
        predictions_path = Path(predictions_path)
    
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    
    with predictions_path.joinpath(signal_level_predictions_name).open('rb') as fin:
        predictions = pickle.load(fin)
    
    stage_dict = {'targets': {}, 'predictions': {}}
    
    for ID in predictions:
        stage_dict['predictions'][ID] = \
        get_epoch_level_predictions_for_pipeline(ID=ID,
                                                 predictions = predictions[ID]['predictions'],
                                                 data_path = stage_path,
                                                 stage_file = stage_file_name,
                                                 apnea_threshold_for_epoch = apnea_threshold_for_epoch,
                                                 sampling_rate = sampling_rate,
                                                 epoch_length = epoch_length)
        stage_dict['targets'][ID] = \
        get_epoch_level_predictions_for_pipeline(ID=ID,
                                                 predictions = predictions[ID]['targets'],
                                                 data_path = stage_path,
                                                 stage_file = stage_file_name,
                                                 apnea_threshold_for_epoch = apnea_threshold_for_epoch,
                                                 sampling_rate = sampling_rate,
                                                 epoch_length = epoch_length)
    
    with save_path.joinpath(save_name).open('wb') as fout:
        pickle.dump(stage_dict,fout)
    
    

def get_epoch_level_predictions_for_pipeline(ID: str,
                                predictions: np.ndarray,
                                data_path: str,
                                stage_file: str,
                                apnea_threshold_for_epoch: float, 
                                sampling_rate: int = 10, 
                                epoch_length: int = 30) -> dict:
    """Takes in ID and array of signal-level predictions  predictions (for sleep 
    epochs only) and output epoch-level predictions for the entire sleep study
    duration.
    INPUTS:
        ID: sleeper ID.
        predictions: 1-D array of predictions (only for sleep epochs)
        data_path: where stage file and apnea/hypopnea targets live
        stage_file: name of staging file
        apnea_threshold_for_epoch: the total length of apneic events required for a
            sleep epoch to be called an apnea epoch
        sampling_rate: sampling_rate at which labels were assigned, typically 10 Hz
        epoch_length: epoch_length in seconds, typically 30 s
    
    RETURNS:
        dictionary, keyed by epoch, with a value of 'A/H' indicating apnea/hypopnea
        and 'None' indicating no apnea/hypopnea
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    
    with data_path.joinpath(stage_file).open('rb') as fs:
        stage_dict = pickle.load(fs)[ID]
    
    if isinstance(predictions, list): predictions = np.array(predictions)

    # Adjust apnea_threshold_for_epoch to be in units of samples
    apnea_threshold_for_epoch *= sampling_rate

    apnea_dict = {epoch: 'None' for epoch in stage_dict}
    counter = 0
    for epoch in sorted(list(stage_dict.keys())):
        if stage_dict[epoch] in ['R','1','2','3']:
            if predictions[counter:counter + sampling_rate*epoch_length].sum() >= apnea_threshold_for_epoch:
                apnea_dict[epoch] = 'A/H'
            counter += sampling_rate*epoch_length
            if counter >= len(predictions):break

    return apnea_dict