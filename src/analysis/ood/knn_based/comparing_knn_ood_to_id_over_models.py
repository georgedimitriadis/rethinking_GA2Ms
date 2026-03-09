

import os

import numpy as np
import pandas as pd

from analysis.ood.knn_based import funcs_on_logs as fl

data_dir = './talent_benchmark/data'

logs_dir = './talent_benchmark/logs'
the_ecmac_ood_log_dir = 'ecmac_2025_11_27_ood_v1'
ecmac_ood_logs_dir = os.path.join(logs_dir, the_ecmac_ood_log_dir)
the_best_ecmac_log_dir = 'ecmac_2025_11_18_fast_v8_splines_ar2'
ecmac_id_logs_dir = os.path.join(logs_dir, the_best_ecmac_log_dir)
the_lightgbm_ood_log_dir = 'lightgbm_2025_11_28_ood_v1'
lightgbm_ood_logs_dir = os.path.join(logs_dir, the_lightgbm_ood_log_dir)

all_log_files_dirs = {'ecmac':[ecmac_ood_logs_dir, ecmac_id_logs_dir], 'lightgbm':[lightgbm_ood_logs_dir]}


def make_df_from_log_files(all_log_files_dirs):
    df = pd.DataFrame()
    dataset_names = fl.get_all_used_dataset_names_from_logs(os.listdir(all_log_files_dirs['ecmac'][0]))

    for model in all_log_files_dirs:
        ood_logs_dir = all_log_files_dirs[model][0]
        ood_log_files = os.listdir(ood_logs_dir)

        if model == 'ecmac':
            id_logs_dir = all_log_files_dirs[model][1]
            id_model_results = fl.get_id_results(id_logs_dir, ood_log_files)
        else:
            id_model_results = fl.get_id_results_from_existing_model(model, dataset_names)

        ood_model_results = fl.get_log_results_from_files(ood_logs_dir, ood_log_files)
        df_model = pd.DataFrame.from_dict(id_model_results, orient='index', columns=[f'{model} ID'])
        #df = pd.concat([df, df_model])
        df[f'{model} ID'] = df_model[f'{model} ID']
        df[f'{model} OOD Avg'] = ''
        df[f'{model} OOD Std Ratio'] = ''
        df[f'{model} Diff'] = ''
        for k, i in enumerate(df.index):
            df.loc[i, f'{model} OOD Avg'] = np.nanmean(ood_model_results[i])
            df.loc[i, f'{model} OOD Std Ratio'] = np.nanstd(ood_model_results[i]) / df.loc[i, f'{model} OOD Avg']
            if df.loc[i, f'{model} OOD Avg'] is not np.nan:
                df.loc[i, f'{model} Diff'] = (df.loc[i, f'{model} OOD Avg'] - df.loc[i, f'{model} ID']) / df.loc[i, f'{model} ID']

    return df


df = make_df_from_log_files(all_log_files_dirs)
df.to_csv('ood_to_id_over_models.csv', index=True)