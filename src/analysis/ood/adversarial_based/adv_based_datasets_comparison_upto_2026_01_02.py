
from collections import OrderedDict
import pandas as pd
import os
from analysis.ood.adversarial_based import funcs_on_logs as fl
from analysis.utils import funcs_maths as fm

experiment_date = '2026_01_02'
experiment_name = 'adv_ood_v9'
models = ['ecmac', 'lightgbm']
epochs = [0, 10]

def ood_data_dir(epoch):
    return os.path.join('talent_benchmark', 'data_ood', 'adversarial_based', experiment_date, str(epoch))

def logs_dir(model, epoch, experiment_date, experiment_name):
    return os.path.join('talent_benchmark','logs',f'{experiment_date}_{model}_{experiment_name}',str(epoch))

logs_folder = logs_dir(models[0], epochs[-1], experiment_date, experiment_name)
used_log_files = os.listdir(logs_folder)
used_datasets = list(fl.get_log_mean_results_from_files(logs_folder, used_log_files).keys())

all_model_dfs = OrderedDict()
for m in models:
    all_model_dfs[m] = pd.DataFrame(index=[fl.get_log_file_details(i)[0] for i in used_log_files if 'txt' in i],
                                      columns=[f'RMSE_{i}' for i in epochs] +[f'R2_{i}' for i in epochs])

for m, model in enumerate(all_model_dfs):
    df = all_model_dfs[model]
    for epoch in epochs:
        logs_folder = logs_dir(models[m], epoch, experiment_date, experiment_name)
        log_files = os.listdir(logs_folder)
        model_results = fl.get_log_mean_results_from_files(logs_folder, log_files)
        for dataset in used_datasets:
            result = model_results[dataset]
            data_dir = ood_data_dir(epoch)
            df.loc[dataset, f'RMSE_{epoch}'] = result[0]
            df.loc[dataset, f'R2_{epoch}'] = fm.r_squared_real_or_pseudo_from_score(data_dir=data_dir, dataset=dataset, scores=result)

epochs_for_dif = [0, 10]
r2_change = pd.DataFrame(index=used_datasets, columns=models + ['model diff'])
for dataset in used_datasets:
    for model in models:
        diffs = all_model_dfs[model].loc[dataset, f'R2_{epochs_for_dif[1]}'] - all_model_dfs[model].loc[dataset, f'R2_{epochs_for_dif[0]}']
        if not ((all_model_dfs[models[0]].loc[dataset, 'R2_0'] <= 0.5 or \
                all_model_dfs[models[1]].loc[dataset, 'R2_0'] <= 0.5) and \
                all_model_dfs[models[0]].loc[dataset, f'R2_{epochs[-1]}'] <= 0):
            r2_change.loc[dataset, model] = diffs
    r2_change.loc[dataset, 'model diff'] = r2_change.loc[dataset, models[0]] - r2_change.loc[dataset, models[1]]

for dataset in used_datasets:
    if (all_model_dfs[models[0]].loc[dataset, 'R2_0'] <= 0.5 or \
        all_model_dfs[models[1]].loc[dataset, 'R2_0'] <= 0.5) and \
        all_model_dfs[models[0]].loc[dataset, f'R2_{epochs[-1]}'] <= 0:
        r2_change = r2_change.drop(index=dataset)