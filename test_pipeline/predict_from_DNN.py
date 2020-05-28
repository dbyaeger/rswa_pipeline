#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:41:57 2020

@author: danielyaeger
"""
import pickle
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from data_generators.rswa_data_generator import DataGeneratorAllWindows

def predict(model_name: str, path_to_model: str, path_to_data: str,
            apnea_dict_name: str, apnea_dict_path: str,
            save_path: str, save_name: str = None,
            verbose: bool = True, evaluate: bool = True):
    """Loads the trials object and finds the model with the lowest loss. Loads
    the corresponding trained model and evaluates it over every model in the 
    test set, saving a dictionary of dictionaries keyed by ID with predictions. 
    """

    # Load the model
    if not isinstance(path_to_model, Path):
        path_to_model = Path(path_to_model)
    
    if not isinstance(apnea_dict_path, Path):
        apnea_dict_path = Path(apnea_dict_path)
    
    apnea_dict_path = apnea_dict_path.joinpath(apnea_dict_name)
    
    model_path = str(path_to_model.joinpath(model_name))
    
    model = load_model(model_path)
    
    # Set up data generator
    test_gen = DataGeneratorAllWindows(path_to_data, apnea_dict_path, 
                                       batch_size=1, window_size=10)        
    IDs = test_gen.list_IDs      
    test_results_dict = {}
    test_label_dict = {}

    evaluations = []
    # iterate over IDs and generate predictions
    for ID in IDs:
        if verbose:
            print(f'ID:\t{ID}')
        
        X, y = test_gen.__getitem_for_ID__(ID)
        y_pred = model.predict(X)
        
        test_results_dict[ID] = y_pred
        test_label_dict[ID] = y

        if evaluate:
          evaluations.append(balanced_accuracy_score(y.argmax(-1),y_pred.argmax(-1)))
    
    if not isinstance(save_path, Path): save_path = Path(save_path)
    
    if save_name is not None:
        save_path = save_path.joinpath(save_name)
    else:
        save_path = save_path.joinpath(f'{model_name}_using_{apnea_dict_name}')
    
    with save_path.open('wb') as fh:
        pickle.dump(test_results_dict, fh)
    
    if evaluate:
        print(f'Mean balanced accuracy at subsequence level: {np.mean(evaluations)} +/- {np.std(evaluations)} for {len(evaluations)} number of subsequences')
      
        sequence_dict_preds = {}
        sequence_dict_truth = {}
        sleeper_IDs = list(set([ID.split('_')[0] for ID in list(test_results_dict.keys())]))
        for ID in sleeper_IDs:
          subseq = -1
          while subseq < 1e3:
            subseq += 1
            try:
              if ID not in sequence_dict_preds:
                sequence_dict_preds[ID] = test_results_dict[f'{ID}_{subseq}']
                sequence_dict_truth[ID] = test_label_dict[f'{ID}_{subseq}']
              else:
                sequence_dict_preds[ID] = np.concatenate((sequence_dict_preds[ID], test_results_dict[f'{ID}_{subseq}']))
                sequence_dict_truth[ID] = np.concatenate((sequence_dict_truth[ID], test_label_dict[f'{ID}_{subseq}']))
            except:
              continue
        
        evaluations = [balanced_accuracy_score(sequence_dict_truth[ID].argmax(-1),sequence_dict_preds[ID].argmax(-1)) \
                       for ID in sequence_dict_preds]
        print(f'Mean balanced accuracy at sleeper level: {np.mean(evaluations)} +/- {np.std(evaluations)} for {len(evaluations)} number of sleepers')
        
        
        
        
    
    
    
    