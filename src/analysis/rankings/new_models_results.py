
import os
import numpy as np
import json
import analysis.rankings.rankings_generator as rp
import analysis.utils.funcs_data_manipulation as fdm

reg_file_name = rp.reg_file_name
bin_file_name = rp.bin_file_name
multi_file_name = rp.multi_file_name


def load_dfs_from_md_files():
    df_reg = rp.generate_df_from_md_data(reg_file_name)
    df_bin = rp.generate_df_from_md_data(bin_file_name)
    df_multi = rp.generate_df_from_md_data(multi_file_name)

    return df_reg, df_bin, df_multi

def get_results_for_dataset(logs_dir, dataset, model):
    results_file = os.path.join(logs_dir, f'{dataset}_{model}.txt')

    with open(results_file, 'rb') as rf:
        lines = [line.decode('utf-8').rstrip() for line in rf]
    if lines[-1] != '--------------------------------------------------':
        return [np.nan], [np.nan]
    for line in lines[-30:]:
        if 'Accuracy Results' in line:
            results = [float(a) for a in line.split(':')[1].split(',')]
        if 'RMSE Results' in line:
            results = [float(a) for a in line.split(':')[1].split(',')]
        if 'Time Results' in line:
            times = [float(a) for a in line.split(':')[1].split(',')]

    return results, times

def pick_df(dataset, df_reg, df_bin, df_multi):
    for i, df in enumerate([df_reg, df_bin, df_multi]):
        types = ['reg', 'bin', 'multi']
        if dataset in df.index:
            return df, types[i]

def fill_dfs_with_model_results(logs_dir, datasets, model):
    df_reg, df_bin, df_multi = load_dfs_from_md_files()

    for dataset in datasets:
        df, dataset_type = pick_df(dataset, df_reg, df_bin, df_multi)
        results, _ = get_results_for_dataset(logs_dir, dataset, model=model)
        if not model in df.columns:
            df[model] = ""
        if dataset_type == 'reg':
            df.loc[dataset, model] = np.mean(results)
        else:
            df.loc[dataset, model] = np.mean(results)

    return df_reg, df_bin, df_multi


'''
r = {df_reg.index[i]: float(df_reg.iloc[i, df_reg.columns.get_loc('mlp')] - df_reg.iloc[i, df_reg.columns.get_loc('ecmac')])/df_reg.iloc[i, df_reg.columns.get_loc('ecmac')] for i in range(len(df_reg))}
b = {df_bin.index[i]: float(df_bin.iloc[i, df_bin.columns.get_loc('ecmac')] - df_bin.iloc[i, df_bin.columns.get_loc('mlp')])/df_bin.iloc[i, df_bin.columns.get_loc('ecmac')] for i in range(len(df_bin))}
m = {df_multi.index[i]: float(df_multi.iloc[i, df_multi.columns.get_loc('ecmac')] - df_multi.iloc[i, df_multi.columns.get_loc('mlp')])/df_multi.iloc[i, df_multi.columns.get_loc('ecmac')] for i in range(len(df_multi))}
'''