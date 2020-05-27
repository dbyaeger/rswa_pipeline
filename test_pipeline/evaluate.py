#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:34:19 2020

@author: danielyaeger
"""

import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import re

def evaluate(ground_truth_name: str, prediction_files: list, data_path: str,
             metrics: list, save_path: str, save_name: str):
    """Evaluates each metric on each file in prediction files"""
    # Set up paths
    if not isinstance(data_path, Path): data_path = Path(data_path)
    
    if not isinstance(save_path, Path): save_path = Path(save_path)
    
    # Get targets
    with data_path.joinpath(ground_truth_name).open('rb') as ft:
        targets = pickle.load(ft)
    
    # Make list of metric names
    metric_names = []
    for metric in metrics:
        metric_name = re.findall(r'function (.*) at', str(metric))[0]
        metric_names.append(f'{metric_name}')
    
    # Dictionary to hold results
    results = {metric_name: [] for metric_name in metric_names}
    results['data'] = []
    
    for prediction in prediction_files:
        results['data'].append(prediction)
        
        with data_path.joinpath(prediction).open('rb') as fin:
            preds = pickle.load(fin)
        
        for metric in metrics:
            scores = []
            for ID in preds:
                scores.append(metric(targets[ID], preds[ID]))
            mean = round(np.mean(scores),3)
            std = round(np.std(scores), 3)
            metric_name = re.findall(r'function (.*) at', str(metric))[0]
            results[metric_name].append(f'{mean} ± {std}')
    
    # Make into a dataframe and save as a .csv file
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(save_path.joinpath(save_name), index = False)
    
    return results_df