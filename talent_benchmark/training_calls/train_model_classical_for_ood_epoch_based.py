import json
import os
import pickle
import importlib.resources as pkg_resources
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
    results_list, time_list, ood_vres_list = [], [], []
    args,default_para,opt_space = get_classical_args()
    metric_name = []
    ood_measure_names = ('OOD_DATA_RMSE', 'OOD_DATA_R2', 'OOD_DATA_OUTER_BOUNDS', 'OOD_DATA_INNER_BOUNDS')

    epoch = default_para[args.model_type]['ood']['epoch'] - 1

    if epoch == -1:
        train_model_classical_main()
    else:
        num_of_reps_to_do = default_para[args.model_type]['ood']['reps']

        with open(os.path.join(args.dataset_path, args.dataset, 'ood_cut_indices.pcl'), 'rb') as f:
            ood_cut_indices = pickle.load(f)
        if num_of_reps_to_do == -1:
            num_of_reps_to_do = len(ood_cut_indices[0])

        info = get_dataset_info(args.dataset_path, args.dataset)

        if args.tune:
            print('======= START TUNING ==========')
            train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path)
            args = tune_hyper_parameters(args, opt_space, train_val_data, info)

            print('======== END TUNING =========')
            print('--------- LEARNING ----------')

        for rep in range(num_of_reps_to_do):
            
            if len([1 for i in ood_cut_indices if i[rep] is not None]) > epoch:
                train_indices = ood_cut_indices[epoch][rep]['train']
                val_indices = ood_cut_indices[epoch][rep]['val']
                test_indices = ood_cut_indices[epoch][rep]['test']
                ood_data_rmse = ood_cut_indices[epoch][rep]['rmse']
                ood_data_r2 = ood_cut_indices[epoch][rep]['r2']
                ood_data_bounds = ood_cut_indices[epoch][rep]['bounds']

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

                        ood_vres_list.append((ood_data_rmse, ood_data_r2, ood_data_bounds[0], ood_data_bounds[1]))
                        method.clear_cache()
                        del method.model
                        del method

        #show_results_classical(args, info, metric_name, results_list, time_list)
        show_results_ood(args, info, metric_name, results_list, time_list, ood_measure_names, ood_vres_list)



