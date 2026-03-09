import numpy as np

from analysis.rankings import new_models_results as nmr


def get_log_file_details(log_file_name):
    """
    Assumes a filename with the structure word_word_word_model.xxx.
    :param log_file_name: The full name of the log file
    :return: Returns the name as word_word_word and the model strings
    """
    parts = log_file_name.split('_')
    dataset_name = ''
    for i, p in enumerate(parts[:-1]):
        dataset_name += p
        if i < len(parts) -2:
            dataset_name += '_'
    model = parts[-1].split('.')[0]

    return dataset_name, model

def get_all_used_dataset_names_from_logs(log_files):
    dataset_names = []
    for log_file in log_files:
        dataset_name, model = get_log_file_details(log_file)
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)

    return dataset_names

def get_model_name_from_log_files(log_files):
    dataset_name, model = get_log_file_details(log_files[0])
    return model

def get_log_mean_results_from_files(log_dir, log_files):
    model_name = get_model_name_from_log_files(log_files)
    dataset_names = get_all_used_dataset_names_from_logs(log_files)
    model_results = {}
    for k, dataset_name in enumerate(dataset_names):
        txt_log_files_of_ds = [i for i in log_files if i==f'{dataset_name}_{model_name}.txt' in i]
        model_results_of_ds = []
        for log_file in txt_log_files_of_ds:
            dataset_name, model = get_log_file_details(log_file)
            results, _ = nmr.get_results_for_dataset(log_dir, dataset_name, model=model_name)
            model_results_of_ds.append(np.mean(results))
        model_results[dataset_name] = model_results_of_ds

    return model_results

def get_id_results(logs_dir, log_files):
    dataset_names = get_all_used_dataset_names_from_logs(log_files)
    dataset_name, model = get_log_file_details(log_files[0])
    model_results = {}
    for dataset_name in dataset_names:
        results, _ = nmr.get_results_for_dataset(logs_dir, dataset_name, model=model)
        model_results[dataset_name] = results[0]

    return model_results

def get_id_results_from_existing_model(model, dataset_names):
    assert model != 'ecmac'
    df_reg, df_bin, df_multi = nmr.load_dfs_from_md_files()
    results = {}
    for name in dataset_names:
        if name in df_reg.index:
            results[name] = df_reg.loc[name, model]
        if name in df_bin.index:
            results[name] = df_bin.loc[name, model]
        if name in df_multi.index:
            results[name] = df_multi.loc[name, model]

    return results