
from collections import OrderedDict
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats

from analysis.ood.adversarial_based import funcs_on_logs as fl
from analysis.utils import funcs_maths as fm


def logs_dir(logs_base_folder, model, r2_quartile):
    return os.path.join(logs_base_folder, model, f'r2_quartile_{r2_quartile}')

knn_data_folder = r'talent_benchmark/data_ood/adversarial_based/2026_01_28'
svn_data_folder = r'talent_benchmark/data_ood/adversarial_based/2026_02_02'
knn_logs_folder = r'talent_benchmark/logs/2026_01_29'
svn_logs_folder = r'talent_benchmark/logs/2026_02_02'

knn_datasets = set(os.listdir(knn_data_folder))
svn_datasets = set(os.listdir(svn_data_folder))
common_datasets = list(knn_datasets & svn_datasets)
all_datasets = list(knn_datasets | svn_datasets)
unique_datasets = list(set(all_datasets) - set(common_datasets))

r2_quartiles = [0, 3]
models = ['ecmac', 'lightgbm']

# Get the minimum and maximum r^2 of the common broken datasets broken with knn and svn
# Then use these to choose for each of these algos which one to choose.
# The logic is: choose knn as default unless the max r^2 of the knn is smaller than 0 and the svn one is larger than 0.
r2s_of_two_algos = pd.DataFrame(columns=['max_svn', 'min_svn', 'max_knn', 'min_knn', 'choice'], index=common_datasets)
for ds in common_datasets:
    knn_res = pd.read_pickle(os.path.join(knn_data_folder, ds, 'ood_results_df.pcl'))
    svn_res = pd.read_pickle(os.path.join(svn_data_folder, ds, 'ood_results_df.pcl'))
    r2s_of_two_algos.loc[ds, 'min_knn'] = knn_res['ood_cut_r2'].min()
    r2s_of_two_algos.loc[ds, 'max_knn'] = knn_res['ood_cut_r2'].max()
    r2s_of_two_algos.loc[ds, 'min_svn'] = svn_res['ood_cut_r2'].min()
    r2s_of_two_algos.loc[ds, 'max_svn'] = svn_res['ood_cut_r2'].max()
    r2s_of_two_algos.loc[ds, 'choice'] = 'knn'
    if r2s_of_two_algos.loc[ds, 'max_knn']>0 and r2s_of_two_algos.loc[ds, 'max_svn']>0:
        r2s_of_two_algos.loc[ds, 'choice'] = 'knn'
    elif r2s_of_two_algos.loc[ds, 'max_knn']<0 and r2s_of_two_algos.loc[ds, 'max_svn']>0:
        r2s_of_two_algos.loc[ds, 'choice'] = 'svn'

# Take out datasets that exist only in one algo that do not end up with a low enough r^2 (-0.1) in the broken data
unique_datasets_pruned = copy(unique_datasets)
for dataset in unique_datasets:
    if dataset in knn_datasets:
        res = pd.read_pickle(os.path.join(knn_data_folder, dataset, 'ood_results_df.pcl'))
    else:
        res = pd.read_pickle(os.path.join(svn_data_folder, dataset, 'ood_results_df.pcl'))
    if res['ood_cut_r2'].min() > 0:
        unique_datasets_pruned.remove(dataset)

all_used_datasets = list(set(unique_datasets_pruned) | set(common_datasets))

# Using the chosen algos (i.e. the logs that come from the broken dataset for the chosen algo) compile all the results in two dfs
all_model_dfs = OrderedDict()
all_model_dfs_last_epoch_datasets = OrderedDict()
for m in models:
    all_model_dfs[m] = pd.DataFrame(index=[fl.get_log_file_details(i)[0] for i in all_used_datasets if 'txt' in i],
                                      columns=[f'RMSE_{i}' for i in r2_quartiles] +[f'R2_{i}' for i in r2_quartiles] +['choice'])

for m, model in enumerate(all_model_dfs):
    df = all_model_dfs[model]
    for r2_quartile in r2_quartiles:
        for dataset in all_used_datasets:
            if dataset in common_datasets:
                used_logs_folder = knn_logs_folder if r2s_of_two_algos.loc[dataset, 'choice'] == 'knn' else svn_logs_folder
                df.loc[dataset,'choice'] = r2s_of_two_algos.loc[dataset, 'choice']
            elif dataset in knn_datasets:
                used_logs_folder = knn_logs_folder
                df.loc[dataset, 'choice'] = 'knn'
            elif dataset in svn_datasets:
                used_logs_folder = svn_logs_folder
                df.loc[dataset, 'choice'] = 'svn'
            logs_folder = logs_dir(used_logs_folder, model, r2_quartile)
            log_files = os.listdir(logs_folder)
            model_results = fl.get_log_mean_results_from_files(logs_folder, log_files)

            data_dir = knn_data_folder if used_logs_folder == knn_logs_folder else svn_data_folder
            result = model_results[dataset]
            df.loc[dataset, f'RMSE_{r2_quartile}'] = result[0]
            df.loc[dataset, f'R2_{r2_quartile}'] = fm.r_squared_real_or_pseudo_from_score(data_dir=data_dir, dataset=dataset, scores=result)



# From the total results make a single df that shows the non-broken data r^2s and the fully broken ones (for r2_quantile=3) for both ecmac and lgbm
two_models_df = pd.DataFrame(index=[i for i in all_used_datasets], columns=[f'{models[0]}_R2 r2_quartile_{r2_quartiles[0]}',
                                                                                   f'{models[1]}_R2 r2_quartile_{r2_quartiles[0]}',
                                                                                   f'{models[0]}_R2 r2_quartile_{r2_quartiles[-1]}',
                                                                                   f'{models[1]}_R2 r2_quartile_{r2_quartiles[-1]}',
                                                                            'choice'])
for dataset in all_used_datasets:
    two_models_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[0]}'] = all_model_dfs[models[0]].loc[dataset, f'R2_{r2_quartiles[0]}'].astype(float)
    two_models_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[0]}'] = all_model_dfs[models[1]].loc[dataset, f'R2_{r2_quartiles[0]}'].astype(float)
    two_models_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[-1]}'] = all_model_dfs[models[0]].loc[dataset, f'R2_{r2_quartiles[-1]}'].astype(float)
    two_models_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[-1]}'] = all_model_dfs[models[1]].loc[dataset, f'R2_{r2_quartiles[-1]}'].astype(float)
    two_models_df.loc[dataset, 'choice'] = all_model_dfs[models[1]].loc[dataset, 'choice']

# Throw away datasets that:
# 1) both have smaller than 0 the results for the 3rd r2_quantile (for the fully broken data)
# 2) both have larger than 0.9 the results for the fully broken data
# 3) both have the results for the unbroken data smaller than 0.5

two_models_pruned_df = copy(two_models_df)
datasets_where_both_fail = []
datasets_where_both_do_great = []
datasets_where_both_cannot_deal_with_data = []
for dataset in all_used_datasets:
    if (two_models_pruned_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[-1]}'] < 0 and \
        two_models_pruned_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[-1]}'] < 0):
        datasets_where_both_fail.append(dataset)
    if (two_models_pruned_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[-1]}'] > 0.9 and \
        two_models_pruned_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[-1]}'] > 0.9):
        datasets_where_both_do_great.append(dataset)
    if (two_models_pruned_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[0]}'] < 0.5 and \
        two_models_pruned_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[0]}'] < 0.5):
        datasets_where_both_cannot_deal_with_data.append(dataset)

for dataset in all_used_datasets:
    if (two_models_pruned_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[-1]}'] < 0 and \
        two_models_pruned_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[-1]}'] < 0) or \
        (two_models_pruned_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[-1]}'] > 0.9 and \
        two_models_pruned_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[-1]}'] > 0.9) or\
        (two_models_pruned_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[0]}'] < 0.5 and \
        two_models_pruned_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[0]}'] < 0.5):
        two_models_pruned_df.drop(dataset, inplace=True)


differences = pd.DataFrame(index=two_models_pruned_df.index, columns=[f'D_{models[0]}', f'D_{models[1]}', 'DD'])
for dataset in differences.index:
    differences.loc[dataset, f'D_{models[0]}'] = two_models_pruned_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[-1]}'] - \
                                                two_models_pruned_df.loc[dataset, f'{models[0]}_R2 r2_quartile_{r2_quartiles[0]}']
    differences.loc[dataset, f'D_{models[1]}'] = two_models_pruned_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[-1]}'] - \
                                                two_models_pruned_df.loc[dataset, f'{models[1]}_R2 r2_quartile_{r2_quartiles[0]}']
    differences.loc[dataset, 'DD'] = differences.loc[dataset, f'D_{models[0]}'] - differences.loc[dataset, f'D_{models[1]}']



# One tailed t-test
t_stat, p_two_tailed = stats.ttest_1samp(differences['DD'].astype(float), 0)
p_one_tailed = p_two_tailed / 2 if t_stat > 0 else 1 - p_two_tailed / 2

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value (one-tailed): {p_one_tailed:.4f}")

# Bootstrap
n_bootstrap = 10000
bootstrap_means = np.array([np.mean(np.random.choice(differences['DD'], size=len(differences['DD']), replace=True))
                            for _ in range(n_bootstrap)])

p_value = np.mean(bootstrap_means <= 0)

print(f"Observed mean: {np.mean(differences['DD']):.4f}")
print(f"Bootstrap p-value (one-tailed): {p_value:.4f}")
print(f"95% CI of mean: [{np.percentile(bootstrap_means, 2.5):.4f}, {np.percentile(bootstrap_means, 97.5):.4f}]")


_ = plt.hist(bootstrap_means)
plt.vlines(x=differences['DD'].mean(), ymin=0, ymax=3000, colors='k', linewidth=2)
plt.vlines(x=bootstrap_means.mean(), ymin=0, ymax=3000, colors='r', linewidth=2)