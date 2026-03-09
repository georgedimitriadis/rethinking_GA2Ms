from analysis.rankings import new_models_results as nmr


def get_log_file_details(log_file_name):
    parts = log_file_name.split('__')
    features_used = int(parts[0].split('_')[1])
    dataset_name = parts[1]
    knn_index = int(parts[2].split('_')[1])
    model = parts[2].split('_')[2].split('.')[0]
    txt_file = True if parts[2].split('.')[1] == 'txt' else False

    return features_used, dataset_name, knn_index, model, txt_file

def get_all_used_dataset_names_from_logs(log_files):
    dataset_names = []
    for log_file in log_files:
        features_used, dataset_name, knn_index, model, txt_file = get_log_file_details(log_file)
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)

    return dataset_names

def get_model_name_from_log_files(log_files):
    features_used, dataset_name, knn_index, model, txt_file = get_log_file_details(log_files[0])
    return model

def get_log_results_from_files(log_dir, log_files):
    model_name = get_model_name_from_log_files(log_files)
    dataset_names = get_all_used_dataset_names_from_logs(log_files)
    model_results = {}
    for k, dataset_name in enumerate(dataset_names):
        txt_log_files_of_ds = [i for i in log_files if f'__{dataset_name}__' in i and 'txt' in i]
        model_results_of_ds = []
        for log_file in txt_log_files_of_ds:
            features_used, _, knn_index, model, txt_file = get_log_file_details(log_file)
            ds_name = f'f_{features_used}__{dataset_name}__c_{knn_index}'
            results, _ = nmr.get_results_for_dataset(log_dir, ds_name, model=model_name)
            model_results_of_ds.append(results[0])
        model_results[dataset_name] = model_results_of_ds

    return model_results

def get_id_results(id_logs_dir, ood_log_files):
    dataset_names = get_all_used_dataset_names_from_logs(ood_log_files)
    features_used, _, knn_index, model, txt_file = get_log_file_details(ood_log_files[0])
    model_results = {}
    for dataset_name in dataset_names:
        results, _ = nmr.get_results_for_dataset(id_logs_dir, dataset_name, model=model)
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