
from collections import OrderedDict

import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt

from analysis.ood.adversarial_based import funcs_on_logs as fl
from analysis.utils import funcs_maths as fm, funcs_data_manipulation as fdm
from analysis.visualisation.data_visualisation_funcs import add_stat_annotation
import seaborn as sb
from scipy import stats

data_experiment_date = '2026_01_28'
logs_experiment_date = '2026_01_29'
models = ['ecmac', 'lightgbm']
r2_quartiles = [0, 3]

def logs_dir(model, r2_quartile, experiment_date):
    return os.path.join('talent_benchmark','logs', experiment_date, model, f'r2_quartile_{r2_quartile}')

data_dir = os.path.join('talent_benchmark', 'data_ood', 'adversarial_based', data_experiment_date)

last_epoch_logs_folder = logs_dir(models[0], r2_quartiles[-1], logs_experiment_date)
last_epoch_used_log_files = os.listdir(last_epoch_logs_folder)
last_epoch_results = fl.get_log_mean_results_from_files(last_epoch_logs_folder, last_epoch_used_log_files)
last_epoch_used_datasets = [i for i in last_epoch_results if not np.isnan(last_epoch_results[i][0])]

zero_epoch_logs_folder = logs_dir(models[0], r2_quartiles[0], logs_experiment_date)
zero_epoch_used_log_files = os.listdir(zero_epoch_logs_folder)
zero_epoch_results = fl.get_log_mean_results_from_files(zero_epoch_logs_folder, zero_epoch_used_log_files)
zero_epoch_used_datasets = [i for i in zero_epoch_results if not np.isnan(zero_epoch_results[i][0])]


all_model_dfs = OrderedDict()
all_model_dfs_last_epoch_datasets = OrderedDict()
for m in models:
    all_model_dfs[m] = pd.DataFrame(index=[fl.get_log_file_details(i)[0] for i in zero_epoch_used_datasets if 'txt' in i],
                                      columns=[f'RMSE_{i}' for i in r2_quartiles] +[f'R2_{i}' for i in r2_quartiles])

for m, model in enumerate(all_model_dfs):
    df = all_model_dfs[model]
    for r2_quartile in r2_quartiles:
        logs_folder = logs_dir(model, r2_quartile, logs_experiment_date)
        log_files = os.listdir(logs_folder)
        model_results = fl.get_log_mean_results_from_files(logs_folder, log_files)
        for dataset in zero_epoch_used_datasets:
            result = model_results[dataset]
            df.loc[dataset, f'RMSE_{r2_quartile}'] = result[0]
            df.loc[dataset, f'R2_{r2_quartile}'] = fm.r_squared_real_or_pseudo_from_score(data_dir=data_dir, dataset=dataset, scores=result)
    all_model_dfs_last_epoch_datasets[model] = df.copy()

for model in all_model_dfs_last_epoch_datasets:
    for dataset in all_model_dfs_last_epoch_datasets[model].index:
        if np.isnan(all_model_dfs_last_epoch_datasets[model].loc[dataset, f'R2_{r2_quartiles[-1]}']):
            all_model_dfs_last_epoch_datasets[model] = all_model_dfs_last_epoch_datasets[model].drop(index=dataset)

# Create a boxplot of the avg r^2 of both models for epochs[0] and epochs[-1]
two_models_df = pd.DataFrame(index=[i for i in last_epoch_used_datasets], columns=[f'{models[0]}_R2 r2_quartile_{r2_quartiles[0]}',
                                                                                   f'{models[1]}_R2 r2_quartile_{r2_quartiles[0]}',
                                                                                   f'{models[0]}_R2 r2_quartile_{r2_quartiles[-1]}',
                                                                                   f'{models[1]}_R2 r2_quartile_{r2_quartiles[-1]}',
                                                                                   'data_type'])
for dataset in last_epoch_used_datasets:
    two_models_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[0]}'] = all_model_dfs[models[0]].loc[dataset, f'R2_{r2_quartiles[0]}'].astype(float)
    two_models_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[0]}'] = all_model_dfs[models[1]].loc[dataset, f'R2_{r2_quartiles[0]}'].astype(float)
    two_models_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[-1]}'] = all_model_dfs[models[0]].loc[dataset, f'R2_{r2_quartiles[-1]}'].astype(float)
    two_models_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[-1]}'] = all_model_dfs[models[1]].loc[dataset, f'R2_{r2_quartiles[-1]}'].astype(float)
    two_models_df.loc[dataset, 'data_type'] = fdm.get_type_of_dataset(data_dir, dataset)

r2_threshold = 0.5
df_to_plot = two_models_df.drop('data_type', axis=1)
df_to_plot = df_to_plot[df_to_plot[f'{models[0]}_R2 r2_quartile_{r2_quartiles[0]}'] > r2_threshold][df_to_plot[f'{models[1]}_R2 r2_quartile_{r2_quartiles[0]}'] >r2_threshold]
for dataset in df_to_plot.index:
    if fdm.get_feature_numbers(data_dir, dataset)['C'] > 0:
        df_to_plot = df_to_plot.drop(index=dataset)
df_to_plot = df_to_plot.astype(float)

f = plt.figure()
ax = f.add_subplot()
ax = sb.boxplot(df_to_plot, ax=ax)
ax.set_xlabel('Model / Epoch')
ax.set_ylabel('R squared')

t_stat, p_val1 = stats.ttest_ind(df_to_plot[f'{models[0]}_R2 r2_quartile_{r2_quartiles[-1]}'], df_to_plot[f'{models[1]}_R2 r2_quartile_{r2_quartiles[-1]}'])
add_stat_annotation(ax, 2, 3, 1 + 0.5 * 2 * 0.1, p_val1)


epochs_for_dif = [r2_quartiles[0], r2_quartiles[1]]
r2_change = pd.DataFrame(index=zero_epoch_used_datasets, columns=['data_type', 'feature_nums'] + [f'{k}_R2_{i}' for k in models for i in epochs_for_dif] + [f'{k}_R2_diff' for k in models ] + ['models_diff'])
for dataset in zero_epoch_used_datasets:
    for model in models:
        diffs = all_model_dfs[model].loc[dataset, f'R2_{epochs_for_dif[1]}'] - all_model_dfs[model].loc[dataset, f'R2_{epochs_for_dif[0]}']
        r2_change.loc[dataset, f'{model}_R2_diff'] = diffs
        for e in epochs_for_dif:
            r2_change.loc[dataset, f'{model}_R2_{e}'] = all_model_dfs[model].loc[dataset, f'R2_{e}']
    r2_change.loc[dataset, 'models_diff'] = r2_change.loc[dataset, f'{models[0]}_R2_diff'] - r2_change.loc[dataset, f'{models[1]}_R2_diff']
    r2_change.loc[dataset, 'data_type'] = fdm.get_type_of_dataset(data_dir, dataset)
    features = fdm.get_feature_numbers(data_dir, dataset)
    r2_change.loc[dataset, 'feature_nums'] = f'N: {features['N']} C: {features['C']}'

for dataset in last_epoch_used_datasets:
    if (all_model_dfs[models[0]].loc[dataset, 'R2_0'] <= 0.5 or \
        all_model_dfs[models[1]].loc[dataset, 'R2_0'] <= 0.5) or \
            (all_model_dfs[models[0]].loc[dataset, f'R2_{r2_quartiles[-1]}'] <= 0.4 and all_model_dfs[models[1]].loc[dataset, f'R2_{r2_quartiles[-1]}'] <= 0.4):
        r2_change = r2_change.drop(index=dataset)


