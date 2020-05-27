#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:55:34 2020

@author: danielyaeger
"""
from pathlib import Path

from test_pipeline.evaluate import evaluate
from test_pipeline.get_full_rswa_predictions import generate_full_rswa_predictions
from test_pipeline.predict_from_DNN import predict

def evaluate_pipeline(prediction_configurations: list,
                      name_of_ground_truth_staging: str,
                      path_to_ground_truth_staging: str,
                      name_of_ground_truth_apnea: str,
                      path_to_ground_truth_apnea: str,
                      path_to_p_files: str,
                      save_path: str,
                      pipeline_results_save_name: str,
                      metrics: list):
    
    # Create full-length targets if they don't exist
    if not isinstance(save_path, Path): save_path = Path(save_path)
    
    if not save_path.joinpath('RSWA_ground_truth_full_targets.p').exists():
        print('Generating full-length RSWA targets')
        generate_full_rswa_predictions(name_of_ground_truth_staging,
                                       path_to_ground_truth_staging,
                                       name_of_ground_truth_apnea,
                                       path_to_ground_truth_apnea,
                                       path_to_p_files,
                                       save_path)
    
    # Generate full-length predictions
    prediction_files = []
    
    for config in prediction_configurations:
        prediction_files.append(config['name_of_predictions_full'])
        
        predict(model_name = config['model_name'],
                path_to_model = config['path_to_model'],
                path_to_data = config['path_to_data'],
                apnea_dict_name = config['apnea_dict_name'],
                apnea_dict_path = config['apnea_dict_path'],
                save_path = config['save_path'],
                save_name = config['name_of_predictions_rem_only'])
        
        generate_full_rswa_predictions(name_of_staging = config['name_of_staging'],
                                       path_to_staging = config['path_to_staging'],
                                       name_of_apnea = config['apnea_dict_name'],
                                       path_to_apnea = config['apnea_dict_path'],
                                       name_of_predictions = config['name_of_predictions_rem_only'],
                                       path_to_predictions = config['save_path'],
                                       save_name = config['name_of_predictions_full'],
                                       save_path = config['save_path'])
    # Get results    
    return evaluate(ground_truth_name = 'RSWA_ground_truth_full_targets.p',
                       prediction_files = prediction_files, 
                       data_path = save_path,
                       metrics = metrics,
                       save_path = save_path,
                       save_name = pipeline_results_save_name)
        
        