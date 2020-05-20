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
            verbose: bool = True):
    """Loads the trials object and finds the model with the lowest loss. Loads
    the corresponding trained model and evaluates it over every model in the 
    test set, saving a dictionary of dictionaries keyed by ID with predictions, 
    true labels, confusion matrix, and balanced accuracy."""

    # Load the model
    if not isinstance(path_to_model, Path):
        path_to_model = Path(path_to_model)
    
    if not isinstance(apnea_dict_path, Path):
        apnea_dict_path = Path(apnea_dict_path)
    
    apnea_dict_path = apnea_dict_path.joinpath(apnea_dict_name)
    
    model_path = str(path_to_model.joinpath(f'model_name'))
    
    model = load_model(model_path)
    
    # Set up data generator
    test_gen = DataGeneratorAllWindows(path_to_data, apnea_dict_path, 
                                       batch_size=8, window_size=10,
                                       channel_list=["Chin", "L Leg", "R Leg"],
                                       n_classes=3, stride=1, shuffle=False,
                                       mode="train")        
    IDs = test_gen.list_IDs      
    test_results_dict = {ID: {} for ID in IDs}
    
    # iterate over IDs and generate predictions
    for ID in IDs:
        if verbose:
            print(f'ID:\t{ID}')
        
        X, y = test_gen._getitem_for_ID__(ID)
        y_pred = model.predict(X)
        
        y = y[:len(y_pred)]
        test_results_dict[ID]['targets'] = y
        test_results_dict[ID]['predictions'] = y_pred
        
        try:
            test_results_dict[ID]['balanaced_accuracy'] = balanced_accuracy_score(y.argmax(-1),
                                                      y_pred.argmax(-1))
        except:
             test_results_dict[ID]['balanaced_accuracy'] = np.nan
        
        test_results_dict[ID]["confusion_matrix"] = confusion_matrix(y.argmax(-1),y_pred.argmax(-1))
        
        if verbose:
            print(f"Balanced accuracy: {test_results_dict[ID]['balanaced_accuracy']}")
    
    if not isinstance(save_path, Path): save_path = Path(save_path)
    
    if save_name is not None:
        save_path = save_path.joinpath(save_name)
    else:
        save_path = save_path.joinpath(f'{model_name}_using_{apnea_dict_name}.p')
    
    with save_path.open('wb') as fh:
        pickle.dump(test_results_dict, fh)

        
        
        
        
        
    
    
    
    