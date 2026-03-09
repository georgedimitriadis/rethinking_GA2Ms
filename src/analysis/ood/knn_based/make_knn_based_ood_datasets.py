
import os
import numpy as np
import argparse
from src.analysis.utils import funcs_knn_ood_data_generation as ogkf, funcs_data_manipulation as dmf


def main(portion_of_data_to_ood, num_of_knn_centers, num_of_combinations, starting_dataset_index):
    data_dir = os.path.join('talent_benchmark', 'data')
    data_dir_ood = os.path.join('talent_benchmark', 'data_ood')
    datasets = ogkf.get_datasets_with_only_numerical_data(data_dir, remove_data_with_nans=True)

    datasets = datasets[starting_dataset_index:]
    for dataset_name in datasets:

        raw_x_data, raw_y_data = dmf.load_nyc_data(data_dir, dataset_name)
        print(f'Doing {dataset_name} with length {len(raw_x_data)} and {raw_x_data.shape[1]} features')

        list_of_non_edge_datapoints_indices = ogkf.get_non_edge_point_indices(raw_x_data,
                                                                             max_allowed_num_of_features_at_edge=1)
        if len(list_of_non_edge_datapoints_indices) < 0.5 * len(raw_x_data):
            list_of_non_edge_datapoints_indices = np.arange(len(raw_x_data))

        array_of_non_edge_datapoints = raw_x_data[list_of_non_edge_datapoints_indices]

        most_central_idx = ogkf.find_point_closer_to_mean(array_of_non_edge_datapoints)
        knn_centers = [most_central_idx]
        if num_of_knn_centers > 1:
            indices_for_centers = list(set(np.arange(len(array_of_non_edge_datapoints))) - set(knn_centers))
            extra_knn_centers = np.random.choice(indices_for_centers, size=num_of_knn_centers-1, replace=False)
            for c in extra_knn_centers:
                knn_centers.append(c)

        updated_portion_of_data_to_ood = portion_of_data_to_ood * len(raw_x_data)/len(list_of_non_edge_datapoints_indices)
        possible_kneighbors_groups, features_used = ogkf.get_knn_based_point_groups(data=array_of_non_edge_datapoints,
                                                                                    portion_of_data_to_ood=updated_portion_of_data_to_ood,
                                                                                    for_points_indices=knn_centers,
                                                                                    num_of_combinations=num_of_combinations)
        for f in features_used.keys():
            for i, feature_comb in enumerate(features_used[f]):
                for c in range(len(features_used[f][feature_comb])):
                    ood_datapoint_indices_from_non_edge = possible_kneighbors_groups[f][feature_comb][c]
                    ood_datapoint_indices_from_raw = np.array(list_of_non_edge_datapoints_indices)[ood_datapoint_indices_from_non_edge]
                    raw_x_data, raw_y_data = dmf.load_nyc_data(data_dir=data_dir, dataset_name=dataset_name)
                    center_idx_on_non_edge_array = features_used[f][feature_comb][c][0]
                    center_idx_on_raw_data = list_of_non_edge_datapoints_indices[center_idx_on_non_edge_array]
                    new_dataset_name = f'f_{raw_x_data.shape[1] - f}__{dataset_name}__c_{c}'
                    data = dmf.generate_new_data_split_deprecated(data=[raw_x_data, raw_y_data],
                                                                  test_indices=ood_datapoint_indices_from_raw)
                    info = dmf.get_dataset_info(data_dir, dataset_name)
                    info['ood_center_point_index'] = int(center_idx_on_raw_data)
                    info['ood_center_point'] = tuple([int(i) for i in raw_x_data[center_idx_on_raw_data]])
                    info['ood_features_used'] = tuple(feature_comb)
                    dmf.save_new_dataset_deprecated(data=data, info=info,
                                                    new_data_dir=data_dir_ood, new_dataset_name=new_dataset_name)

        print(f' --- Done {dataset_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--percent_ood", type=float, default=0.1)
    parser.add_argument("-c", "--num_center_points", type=int, default=2)
    parser.add_argument("-s", "--starting_dataset_index", type=int, default=0)
    parser.add_argument("-b", "--num_combinations", type=int, default=2)
    args = parser.parse_args()

    main(portion_of_data_to_ood=args.percent_ood,
         num_of_knn_centers=args.num_center_points,
         starting_dataset_index=args.starting_dataset_index,
         num_of_combinations=args.num_combinations)
