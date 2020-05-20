#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:08:57 2020

@author: danielyaeger
"""
import numpy as np
from pathlib import Path
import pickle

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
    # Create signal-level and epoch-level representations
    for epoch in sorted(list(stage_dict.keys())):
        if stage_dict[epoch] in ['R','1','2','3']:
            if predictions[counter:counter + sampling_rate*epoch_length].sum() >= apnea_threshold_for_epoch:
                apnea_dict[epoch] = 'A/H'
            counter += sampling_rate*epoch_length
            if counter >= len(predictions):break

    return apnea_dict