
import argparse
import os
import numpy as np
import pandas as pd

from analysis.ood.adversarial_based.adv_based_ood_data_generator_funcs import (cma_rectangle_search, data_norm)
from analysis.utils import funcs_data_manipulation as dmf
import pickle

def generate_train_and_val_indices(all_results, data_size, k_ratio):

    all_epochs_test_indices = [res[4] for res in all_results]

    all_epochs_train_val_indices = [list(set(range(data_size)) - set(test_indices)) for test_indices in
                                    all_epochs_test_indices]
    all_epochs_train_indices = [np.random.choice(train_val_indices, replace=False,
                                                 size=int(k_ratio * len(all_epochs_train_val_indices[0]))).tolist() for
                                train_val_indices in all_epochs_train_val_indices]
    all_epochs_val_indices = [list(set(all_epochs_train_val_indices[i]) - set(all_epochs_train_indices[i])) for i in
                              range(len(all_results))]

    return all_epochs_train_indices, all_epochs_val_indices, all_epochs_test_indices

def add_cut_results_to_2d_train_val_test_indices_list(all_results, all_epochs_train_indices, all_epochs_test_indices,
                                                      all_epochs_val_indices, ood_cut_indices, rep):
    all_epochs_rmses = [res[1] for res in all_results]
    all_epochs_r2s = [res[2] for res in all_results]
    all_epochs_bounds = [res[3] for res in all_results]
    for epoch in range(len(all_results)):
        ood_cut_indices[epoch][rep] = {'train': all_epochs_train_indices[epoch], 'val': all_epochs_val_indices[epoch],
                                 'test': all_epochs_test_indices[epoch], 'r2': all_epochs_r2s[epoch],
                                 'rmse': all_epochs_rmses[epoch], 'bounds': all_epochs_bounds[epoch]}

    return ood_cut_indices

def create_empty_2d_list(num_of_worsening_sets, num_of_reps):
    ood_cut_indices = []
    for s in range(num_of_worsening_sets):
        ood_cut_indices.append([])
        for r in range(num_of_reps):
            ood_cut_indices[-1].append(None)
    return ood_cut_indices

def add_to_the_results_dataframe(ood_results_df, all_results, all_epochs_train_indices, all_epochs_val_indices, rep):
    for i, result in enumerate(all_results):
        temp_dict = {'repetition': rep, 'epoch': i, 'test_indices': result[4], 'train_indices': all_epochs_train_indices[i],
                     'val_indices': all_epochs_val_indices[i], 'ood_cut_rmse': result[1],
                     'ood_cut_r2': result[2], 'ood_cut_upper_bounds': result[3][1], 'ood_cut_lower_bounds': result[3][0]}

        ood_results_df.loc[len(ood_results_df)] = temp_dict
    return ood_results_df

def main(dataset_name, data_dir, base_data_ood_dir, search_iters, k_ratio, num_of_worsening_sets, num_of_reps, use_knr):
    if use_knr == 'False' or use_knr == 'false' or use_knr == '0' or use_knr == 0:
        use_knr = False
    else:
        use_knr = True

    num_samples = dmf.get_total_sample_size(data_dir, dataset_name)
    features = dmf.get_feature_numbers(data_dir, dataset_name)
    total_features = features['total']
    categorical_features = features['C']
    dataset_type = dmf.get_type_of_dataset(data_dir, dataset_name)
    ood_results_df = pd.DataFrame(columns=['repetition', 'epoch', 'train_indices', 'val_indices', 'test_indices',
                                           'ood_cut_rmse', 'ood_cut_r2',
                                           'ood_cut_lower_bounds', 'ood_cut_upper_bounds'])
    if total_features < 31 and dataset_type == 'reg' and categorical_features == 0:
        x_data_preproc, y_data_preproc = dmf.load_preprocessed_xy_data(dataset_name, data_dir)
        _, unnormalised_y_data, _ = dmf.load_nyc_data(data_dir, dataset_name)

        print(f'Doing  -- {dataset_name} -- with length = {len(x_data_preproc)}  and features = {x_data_preproc.shape[1]}')

        norm_n_data = data_norm(x_data_preproc)

        # Create the new dataset folder
        ood_dataset_path = os.path.join(base_data_ood_dir, dataset_name)
        if not os.path.isdir(ood_dataset_path):
            os.mkdir(ood_dataset_path)

        # Copy the normal data into it
        names = ['N_train', 'N_val', 'N_test', 'C_train', 'C_val', 'C_test', 'y_train', 'y_val', 'y_test']
        dmf.transfer_info_json(os.path.join(data_dir, dataset_name), ood_dataset_path, iteration=None)
        data = dmf.load_ncy_train_val_test_data(data_dir, dataset_name)
        dmf.save_data(data, names, ood_dataset_path)

        # Run the cma_rectangle_search code and save the results for all epochs and all repetitions in the dataset folder
        ood_cut_indices = create_empty_2d_list(num_of_worsening_sets, num_of_reps)
        min_num_epochs = num_of_worsening_sets
        max_num_epocs = 0
        for r in range(num_of_reps):
            all_results = cma_rectangle_search(X_norm=norm_n_data, X_raw=x_data_preproc, y=unnormalised_y_data,
                                               k_ratio=k_ratio, iters=search_iters,
                                               max_num_of_mse_drops=num_of_worsening_sets, return_full_history=True,
                                               dataset_name=dataset_name, use_knr= use_knr)

            if all_results is None:
                print(f'The dataset {dataset_name} added changed test indices sets TOO SLOWLY. Aborting.')
                break

            else:
                all_epochs_train_indices, all_epochs_val_indices, all_epochs_test_indices = \
                    generate_train_and_val_indices(all_results, data_size=len(norm_n_data), k_ratio=k_ratio)
                ood_cut_indices = add_cut_results_to_2d_train_val_test_indices_list(all_results, all_epochs_train_indices,
                                                                                    all_epochs_test_indices,
                                                                                    all_epochs_val_indices,
                                                                                    ood_cut_indices, rep=r)
                ood_results_df = add_to_the_results_dataframe(ood_results_df, all_results,
                                                              all_epochs_train_indices, all_epochs_val_indices, rep=r)

                if min_num_epochs > len(all_results):
                    min_num_epochs = len(all_results)
                if max_num_epocs < len(all_results):
                    max_num_epocs = len(all_results)

        with open(os.path.join(ood_dataset_path, 'ood_cut_indices.pcl'), 'wb') as f:
            pickle.dump(ood_cut_indices, f)
        ood_results_df.to_pickle(os.path.join(ood_dataset_path, 'ood_results_df.pcl'))

        print(f'FINISHED {dataset_name}. FOUND BETWEEN {min_num_epochs} AND {max_num_epocs} OOD DATASETS.')
    else:
        print(f'{dataset_name.upper()} IS EITHER TOO BIG OR NOT INPUT AND OUTPUT CONTINUOUS.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--dataset", type=str)
    parser.add_argument("-d", "--dataset_path", type=str, default='data')
    parser.add_argument("-o", "--ood_dataset_path", type=str, default='data_ood',
                        help='This is the full path into which the individual dataset folders will be stored.')
    parser.add_argument("-i", "--rect_search_iters", type=int, default=100)
    parser.add_argument("-k", "--k_ratio", type=float, default=0.9)
    parser.add_argument("-r", "--num_of_repetitions", type=int, default=1)
    parser.add_argument("-w", "--num_of_worsening_sets", type=int, default=10)
    parser.add_argument("-u", "--use_knr", default='True', type=str)
    args = parser.parse_args()

    main(dataset_name=args.dataset, data_dir=args.dataset_path, base_data_ood_dir=args.ood_dataset_path, search_iters=args.rect_search_iters,
         k_ratio=args.k_ratio, num_of_worsening_sets=args.num_of_worsening_sets, num_of_reps=args.num_of_repetitions,
         use_knr=args.use_knr)


