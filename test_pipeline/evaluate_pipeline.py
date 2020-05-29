#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:55:34 2020

@author: danielyaeger
"""
from pathlib import Path

from test_pipeline.evaluate import evaluate
from test_pipeline.generate_rswa_targets import generate_targets
from test_pipeline.predict_from_DNN import predict
from test_pipeline.get_full_rswa_predictions import generate_full_rswa_predictions

def evaluate_pipeline(prediction_configurations: list,
                      name_of_ground_truth_staging: str,
                      path_to_ground_truth_staging: str,
                      name_of_ground_truth_apnea: str,
                      path_to_ground_truth_apnea: str,
                      path_to_p_files: str,
                      save_path: str,
                      pipeline_results_save_name: str,
                      metrics: list,
                      replace_files: bool = False):
    
    # Create full-length targets if they don't exist
    if not isinstance(save_path, Path): save_path = Path(save_path)
    
    if not save_path.joinpath('RSWA_ground_truth_full_targets.p').exists():
        print('Generating full-length RSWA targets')
        generate_targets(name_of_ground_truth_staging,
                                       path_to_ground_truth_staging,
                                       name_of_ground_truth_apnea,
                                       path_to_ground_truth_apnea,
                                       path_to_p_files,
                                       save_path)
    
    # Generate full-length predictions
    prediction_files = []
    
    for config in prediction_configurations:
        prediction_files.append(config['name_of_predictions_full'])
        
        if not Path(config['save_path']).joinpath(config['name_of_predictions_rem_only']).exists() or replace_files:
            predict(model_name = config['model_name'],
                    path_to_model = config['path_to_model'],
                    path_to_data = config['path_to_data'],
                    apnea_dict_name = config['apnea_dict_name'],
                    apnea_dict_path = config['apnea_dict_path'],
                    save_path = config['save_path'],
                    save_name = config['name_of_predictions_rem_only'])
        
        if not Path(config['save_path']).joinpath(config['name_of_predictions_full']).exists() or replace_files:
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

if __name__ == "__main__":
    
    from sklearn.metrics import balanced_accuracy_score
    
    # Set up paths and 
    model_path = Path('/content/gdrive/My Drive/').joinpath('Models')
    path_to_apneas = Path('/content/gdrive/My Drive/').joinpath('Apnea_Dicts')
    prediction_configurations = [
                      {'model_name': 'resnet_no_apnea_multi_channel_with_artifact_reduction_window_10.hdf5',
                       'path_to_model': model_path,
                       'path_to_data': Path('/content/gdrive/My Drive/').joinpath('Data/artifact_reduced_emg2'),
                       'apnea_dict_name': 'ground_truth_apnea_dict.p',
                       'apnea_dict_path': path_to_apneas,
                       'name_of_predictions_rem_only': 'REM_human_staging_human_apnea_ml_rswa_AR.p',
                       'save_path': Path('/content/gdrive/My Drive/').joinpath('RSWA_Predictions'),
                       'name_of_staging': 'stage_dict.p',
                       'path_to_staging': Path('/content/gdrive/My Drive/').joinpath('Data/raw_no_baseline_all'),
                       'name_of_predictions_full': 'FULL_human_staging_human_apnea_ml_rswa_AR.p'
                       },
                      {'model_name': 'resnet_no_apnea_multi_channel_window_10.hdf5',
                       'path_to_model': model_path,
                       'path_to_data': Path('/content/gdrive/My Drive/').joinpath('Data/NO_artifact_reduced_emg2'),
                       'apnea_dict_name': 'ground_truth_apnea_dict.p',
                       'apnea_dict_path': path_to_apneas,
                       'name_of_predictions_rem_only': 'REM_human_staging_human_apnea_ml_rswa.p',
                       'save_path': Path('/content/gdrive/My Drive/').joinpath('RSWA_Predictions'),
                       'name_of_staging': 'stage_dict.p',
                       'path_to_staging': Path('/content/gdrive/My Drive/').joinpath('Data/raw_no_baseline_all'),
                       'name_of_predictions_full': 'FULL_human_staging_human_apnea_ml_rswa.p'
                       },
                       {'model_name': 'resnet_no_apnea_multi_channel_with_artifact_reduction_window_10.hdf5',
                       'path_to_model': model_path,
                       'path_to_data': Path('/content/gdrive/My Drive/').joinpath('Data/artifact_reduced_emg2'),
                       'apnea_dict_name': 'human_stage_five_conv_two_dense_epoch.p',
                       'apnea_dict_path': path_to_apneas,
                       'name_of_predictions_rem_only': 'REM_human_staging_ML_apnea_ML_rswa_AR.p',
                       'save_path': Path('/content/gdrive/My Drive/').joinpath('RSWA_Predictions'),
                       'name_of_staging': 'ground_truth_stage_dict.p',
                       'path_to_staging': Path('/content/gdrive/My Drive/').joinpath('Stage_Dicts'),
                       'name_of_predictions_full': 'FULL_human_staging_ML_apnea_ML_rswa_AR.p'
                       },
                      {'model_name': 'resnet_no_apnea_multi_channel_window_10.hdf5',
                       'path_to_model': model_path,
                       'path_to_data': Path('/content/gdrive/My Drive/').joinpath('Data/NO_artifact_reduced_emg2'),
                       'apnea_dict_name': 'human_stage_five_conv_two_dense_epoch.p',
                       'apnea_dict_path': path_to_apneas,
                       'name_of_predictions_rem_only': 'REM_human_staging_ML_apnea_ML_rswa.p',
                       'save_path': Path('/content/gdrive/My Drive/').joinpath('RSWA_Predictions'),
                       'name_of_staging': 'ground_truth_stage_dict.p',
                       'path_to_staging': Path('/content/gdrive/My Drive/').joinpath('Stage_Dicts'),
                       'name_of_predictions_full': 'FULL_human_staging_ML_apnea_ML_rswa.p'
                       }
                       ]
    
    evaluate_pipeline(prediction_configurations = prediction_configurations,
                      name_of_ground_truth_staging = 'ground_truth_stage_dict.p',
                      path_to_ground_truth_staging = Path('/content/gdrive/My Drive/').joinpath('Stage_Dicts'),
                      name_of_ground_truth_apnea = 'ground_truth_apnea_dict.p',
                      path_to_ground_truth_apnea = Path('/content/gdrive/My Drive/').joinpath('Apnea_Dicts'),
                      path_to_p_files = Path('/content/gdrive/My Drive/').joinpath('Data/artifact_reduced_emg2'),
                      save_path = Path('/content/gdrive/My Drive/').joinpath('RSWA_Predictions'),
                      pipeline_results_save_name = 'full_pipeline_results.csv',
                      metrics =  [balanced_accuracy_score],
                      replace_files=True)
        
        