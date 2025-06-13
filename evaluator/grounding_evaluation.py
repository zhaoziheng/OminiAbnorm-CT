import os
import torch
import numpy as np
import pandas as pd
import json
from medpy import metric
from .SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_detection_ratio, compute_sensitivity, compute_specificity

def calculate_metric_percase(pred, gt, dice=True, nsd=True, se=True, sp=True, dr=True, success_threshold_ls=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    assert dice==True or nsd==True, 'At least one of dice or nsd should be calculated'
          
    metrics = {}
          
    if np.sum(gt) == 0.0:
        if np.sum(pred) == 0.0:
            if dice:
                metrics['dice'] = 1.0
            if nsd:
                metrics['nsd'] = 1.0
            if se:
                metrics['se'] = 1.0
            if sp:
                metrics['sp'] = 1.0
            if dr:
                for success_threshold in success_threshold_ls:
                    metrics[f'dr@{success_threshold}'] = 1.0
        else:
            if dice:
                metrics['dice'] = 0.0
            if nsd:
                metrics['nsd'] = 0.0
            if se:
                metrics['se'] = 0.0
            if sp:
                metrics['sp'] = 0.0
            if dr:
                for success_threshold in success_threshold_ls:
                    metrics[f'dr@{success_threshold}'] = 0.0
        return metrics
        
    if dice:
        dice_score = metric.binary.dc(pred, gt)
        metrics['dice'] = dice_score
    
    if nsd:
        surface_distances = compute_surface_distances(gt, pred, [1, 1])
        nsd_score = compute_surface_dice_at_tolerance(surface_distances, 1)
        metrics['nsd'] = nsd_score
        
    if se:
        se_score = compute_sensitivity(gt, pred)
        metrics['se'] = se_score
    
    if sp:
        sp_score = compute_specificity(gt, pred)
        metrics['sp'] = sp_score
        
    if dr:
        for success_threshold in success_threshold_ls:
            metrics[f'dr@{success_threshold}'] = compute_detection_ratio(gt, pred, success_threshold)
    
    return metrics

def statistic_results(results_of_samples, csv_path):
    
    # datasets --> labels --> metrics
    datasets_labels_metrics = {}   # {'DeepLesion':{'':{'dice':[0.8, 0.9, ...], ...} ...}, ...}

    # datasets --> samples --> labels --> metrics
    samples_labels_metrics = {}   # {'DeepLesion':{'0':{'':{'dice':0.8, ...} ...}, ...} 记录每个dataset里的sample（行）
    
    # datsets --> labels
    datasets_labels_sets = {}    # {'DeepLesion':set(''), ...}  记录每个dataset里的label种类（列）         
    
    # Get statistic and record
    
    for dataset_name, sample_id, scores, label_ls, description_ls, type_ls_ls in results_of_samples:
        dataset_name = f'{dataset_name}'
        
        if dataset_name not in datasets_labels_metrics:
            datasets_labels_metrics[dataset_name] = {}  # {'DeepLesion':{'type1':{'dice':[0.8, 0.9, ...], 'nsd':[0.8, 0.9, ...]} 'other_type':...}, 'Radiopedia':...}
        if dataset_name not in datasets_labels_sets:
            datasets_labels_sets[dataset_name] = set()  # {'DeepLesion':set('type1', 'type2', ...)}
        if dataset_name not in samples_labels_metrics:
            samples_labels_metrics[dataset_name] = {}
        samples_labels_metrics[dataset_name][sample_id] = {}   # {'DeepLesion':{'0':{}}}
        
        for metric_dict, type_ls in zip(scores, type_ls_ls):
            # accumulate metrics （for per dataset per class
            # datasets_labels_metrics = {'DeepLesion':{'type1':{'dice':[0.8, 0.9, ...], 'nsd':[0.8, 0.9, ...]} 'other_type':...}, 'Radiopedia':...}
            for type_txt in type_ls:
                if type_txt not in datasets_labels_metrics[dataset_name]:
                    datasets_labels_metrics[dataset_name][type_txt] = {k:[v] for k,v in metric_dict.items()}
                else:
                    for k,v in metric_dict.items():
                        datasets_labels_metrics[dataset_name][type_txt][k].append(v)
            
                # statistic labels
                # datasets_labels_sets = {'DeepLesion':set('', ...)}
                if type_txt not in datasets_labels_sets[dataset_name]:
                    datasets_labels_sets[dataset_name].add(type_txt)
            
                # record metrics （for per dataset per sample per class
                # samples_labels_metrics = {'DeepLesion':{'0':{'type1':{'dice':0.8, 'nsd':0.9}, 'type2':...}, 'sample2':...}, 'Radiopedia':...}
                samples_labels_metrics[dataset_name][sample_id][type_txt] = {k:v for k,v in metric_dict.items()}
                
    # average and log (列为metrics，例如dice，nsd...)
    # create a df like:
    # {
    #   'DeepLesion': [0.xx, 0.xx, ...]    # 每个sample下所有type平均得到sample的metric，然后所有sample平均得dataset的type  # 在T之前，这是一列
    #   'DeepLesion, type1': [0.68, 0.72, ...]  # 每个数据集每个type内部的所有sample平均
    #   ...
    # }
    avg_df = {}

    # datasets_labels_metrics = {'DeepLesion':{'type1':{'dice':[0.8, 0.9, ...], 'nsd':[0.8, 0.9, ...]} 'other_type':...}, 'Radiopedia':...}
    for dataset in datasets_labels_metrics.keys():
        # avg over each samples under each dataset&type
        for label in datasets_labels_metrics[dataset].keys():
            avg_df[f'{dataset} | {label}'] = []
            for metric in datasets_labels_metrics[dataset][label].keys():
                label_metric = np.average(datasets_labels_metrics[dataset][label][metric])
                avg_df[f'{dataset} | {label}'].append(label_metric)  # 'DeepLesion, type1': [0.68, 0.72] list of num_metrics
                
        # Calculate average across all categories/types for each dataset
        avg_df[dataset+'(category_avg)'] = []
        # Get all metrics from the first label to know what metrics we're averaging
        sample_metrics = list(datasets_labels_metrics[dataset][list(datasets_labels_metrics[dataset].keys())[0]].keys())
        for metric in sample_metrics:
            all_values = []
            for label in datasets_labels_metrics[dataset].keys():
                all_values.append(np.average(datasets_labels_metrics[dataset][label][metric]))  # avg over all samples under this category
            category_avg = np.average(all_values) if all_values else 0.0    # avg over all categories for this metric
            avg_df[dataset+'(category_avg)'].append(category_avg)

    # by defult, print the dice (1st metric) of each dataset 
    info = 'Metrics of Each Dataset:\n'
    # samples_labels_metrics = {'DeepLesion':{'0':{'type1':{'dice':0.8, 'nsd':0.9, ...}, 'type2':...}, 'sample2':...}, 'Radiopedia':...}         
    for dataset in samples_labels_metrics.keys():
        # avg over each samples under each dataset
        avg_df[dataset+'(sample_avg)'] = {k:[] for k in metric_dict.keys()}    # 'DeepLesion': {'dice':[0.8, ...] 'nsd':[0.5, ...], ...}
        for sample_id, label_metrics in samples_labels_metrics[dataset].items():
            # first avg over each type under each sample as the score for this sample
            sample_avg_metrics = {} # {'dice':[0.8, 0.9, ...], 'nsd':[0.8, 0.9, ...] ...}
            for label, metric_dict in label_metrics.items():
                for metric, value in metric_dict.items():
                    if metric not in sample_avg_metrics:
                        sample_avg_metrics[metric] = [value]
                    else:
                        sample_avg_metrics[metric].append(value)
            sample_avg_metrics = {k:np.average(v) for k,v in sample_avg_metrics.items()} # {'dice':0.8, 'nsd':0.9, ...}
            for metric, value in sample_avg_metrics.items():
                avg_df[dataset+'(sample_avg)'][metric].append(value)
        # average over each sample as the score for this dataset
        avg_df[dataset+'(sample_avg)'] = {k:np.average(v) for k,v in avg_df[dataset+'(sample_avg)'].items()} # 'DeepLesion': {'dice':0.x, 'nsd':0.x, ...}
        
        info += f'{dataset}(sample_avg)  |  ' 
        for k ,v in avg_df[dataset+'(sample_avg)'].items():
            info += f'{v}({k})  |  '
        info += '\n'
        avg_df[dataset+'(sample_avg)'] = list(avg_df[dataset+'(sample_avg)'].values())
        
    print(info)
    avg_df = pd.DataFrame(avg_df).T
    avg_df.columns = list(metric_dict.keys())   # ['dice', 'nsd']
    avg_df.to_csv(csv_path, encoding='utf-8')        

    # detailed log
    # multi-sheet, two for each dataset
    df_list = [['summary', avg_df]]
    for dataset, label_set in datasets_labels_sets.items():
        metric_df ={}
        metric_df['dice'] = {}
        metric_df['se'] = {}
        metric_df['sp'] = {}
        
        # samples_labels_metrics = {'DeepLesion':{'sample1':{'type1':{'dice':0.8, ...} 'type2':...}, 'sample2':...}
        for image_id, label_dict in samples_labels_metrics[dataset].items():
            for metric in metric_df:
                tmp = []    # one dice for each label in this dataset
                for label in label_set:
                    score = label_dict[label][metric] if label in label_dict else ''
                    tmp.append(score)
                metric_df[metric][image_id] = tmp   
        
        for metric, metric_df in metric_df.items():
            metric_df = pd.DataFrame(metric_df).T
            metric_df.columns = list(label_set)
            df_list.append([dataset+f'({metric})', metric_df])
        
    xlsx_path = csv_path.replace('.csv', '.xlsx')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for name, df in df_list:
            # 将每个 DataFrame 写入一个 sheet(sheet name must be < 31)
            if len(name) > 31:
                name = name[len(name)-31:]
            df.to_excel(writer, sheet_name=name, index=True)