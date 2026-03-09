import json
import os
import pickle
import importlib.resources as pkg_resources

import numpy as np

import TALENT
from tqdm import tqdm

from TALENT.model.lib.data import get_dataset
from TALENT.model.utils import (
    get_classical_args, tune_hyper_parameters,
    show_results_classical, get_method, set_seeds, show_results_ood
)
from training_calls.train_model_classical import main as train_model_classical_main

from analysis.utils.funcs_data_manipulation import generate_new_data_split_from_train_val_test_indices, \
    get_dataset_info, generate_trainval_test_structure_from_individual_datasets

if __name__ == '__main__':
    results_list, time_list = [], []
    args,default_para,opt_space = get_classical_args()

    r2_quartile = default_para[args.model_type]['ood']['r2_quartile']

    if r2_quartile == 0:
        train_model_classical_main()
    else:
        with open(os.path.join(args.dataset_path, args.dataset, 'ood_results_df.pcl'), 'rb') as f:
            ood_results_df = pickle.load(f)

        info = get_dataset_info(args.dataset_path, args.dataset)

        r = ood_results_df['ood_cut_r2']
        ranges = [r.min() + i * (r.max() - r.min()) / 3 for i in [3, 2, 1]] + [r.min()]
        r2_range = [ranges[r2_quartile-1], ranges[r2_quartile]]
        selected_indices = ood_results_df[(ood_results_df['ood_cut_r2']>=r2_range[1]) & (ood_results_df['ood_cut_r2']<=r2_range[0])].index

        for i in selected_indices:

            train_indices = ood_results_df.at[i, 'train_indices']
            val_indices = ood_results_df.at[i, 'val_indices']
            test_indices = ood_results_df.at[i, 'test_indices']

            if train_indices is not None and test_indices is not None and val_indices is not None:
                n_train, n_val, n_test, c_train, c_val, c_test, y_train, y_val, y_test = \
                    generate_new_data_split_from_train_val_test_indices(args.dataset, args.dataset_path, train_indices,
                                                                        val_indices, test_indices)

                train_val_data, test_data = \
                    generate_trainval_test_structure_from_individual_datasets(n_train, n_val, n_test,
                                                                              c_train, c_val,c_test,
                                                                              y_train, y_val, y_test)
                # if args.tune:
                #   args = tune_hyper_parameters(args, opt_space, train_val_data, info)

                for seed in tqdm(range(args.seed_num)):
                    args.seed = seed  # update seed
                    set_seeds(args.seed)
                    method = get_method(args.model_type)(args, info['task_type'] == 'regression')
                    time_cost = method.fit(train_val_data, info, train=True)
                    vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)

                    results_list.append(vres)
                    time_list.append(time_cost)

                    method.clear_cache()
                    del method.model
                    del method

        show_results_classical(args, info, metric_name, results_list, time_list)
        #show_results_ood(args, info, metric_name, results_list, time_list, ood_measure_names, ood_vres_list)



