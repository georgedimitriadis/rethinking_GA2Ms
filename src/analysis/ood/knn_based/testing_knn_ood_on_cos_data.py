
import os
import numpy as np
from src.analysis.utils import funcs_knn_ood_data_generation as ogkf, funcs_data_manipulation as dmf
from analysis.ood.knn_based.loaddata import get_data
from src.analysis.visualisation import data_visualisation_funcs as dvf


N_SAMPLES = 12000
datasets = get_data(n_samples=N_SAMPLES)

#ax = plt.figure().add_subplot(projection='3d')
#ax.scatter(xs=datasets[0][0][:, 0], ys=datasets[0][0][:, 1], zs=datasets[0][1])

cosX1cosX2_X = datasets[0][0]
cosX1cosX2_Y = datasets[0][1].reshape(-1, 1)

num_of_features = cosX1cosX2_X.shape[1]
features_names = dmf.get_features_names(None, None, num_of_features)
features_ranges = list(zip(np.min(cosX1cosX2_X, axis=0), np.max(cosX1cosX2_X, axis=0)))


list_of_non_edge_datapoints_indices = ogkf.get_non_edge_point_indices(cosX1cosX2_X,
                                                                     max_allowed_num_of_features_at_edge=1)
array_of_non_edge_datapoints = cosX1cosX2_X[list_of_non_edge_datapoints_indices]

most_central_idx = ogkf.find_point_closer_to_mean(array_of_non_edge_datapoints)

portion_of_data_to_ood = 0.1 * len(cosX1cosX2_X)/len(list_of_non_edge_datapoints_indices)
for_points_indices = [most_central_idx]
all_possible_kneighbors_groups, features_used = ogkf.get_knn_based_point_groups(data=array_of_non_edge_datapoints,
                                                                                portion_of_data_to_ood=portion_of_data_to_ood,
                                                                                for_points_indices=for_points_indices)


#a_random_set_of_points = np.array(list(all_possible_kneighbors_groups[0]))[0]
#dvf.show_2d_combinations([cosX1cosX2_X, array_of_non_edge_datapoints[a_random_set_of_points], array_of_non_edge_datapoints[most_central_idx].reshape(1,-1)], features_names)

data_dir_ood = os.path.join('talent_benchmark', 'data_ood', 'cosX1cosX2')
dataset_name = 'cosX1cosX2'

info = {"task_type": "regression",
        "n_num_features": 2,
        "n_cat_features": 0,
        "num_feature_intro": {"X0": "X0", "X1": "X1"}}

for f in features_used.keys():
    for i, feature_comb in enumerate(features_used[f]):
        ood_datapoint_indices_from_non_edge = all_possible_kneighbors_groups[f][i]
        ood_datapoint_indices_from_raw = np.array(list_of_non_edge_datapoints_indices)[ood_datapoint_indices_from_non_edge]
        new_dataset_name = f'{dataset_name}_{str(feature_comb)[1:-1].replace(", ", "_")}_OOD'
        data = dmf.generate_new_data_split_deprecated(data=[cosX1cosX2_X, cosX1cosX2_Y],
                                                      test_indices=ood_datapoint_indices_from_raw)
        info['train_size'] = len(data[0])
        info['test_size'] = len(data[1])
        info['val_size'] = len(data[2])
        dmf.save_new_dataset_deprecated(data=data, info=info,
                                        new_data_dir=os.path.join(data_dir_ood, dataset_name), new_dataset_name=new_dataset_name)


x_tr = np.load(os.path.join('talent_benchmark/data_ood/cosX1cosX2_0_1_OOD', 'N_train.npy'))
x_te = np.load(os.path.join('talent_benchmark/data_ood/cosX1cosX2_0_1_OOD', 'N_test.npy'))
x_val = np.load(os.path.join('talent_benchmark/data_ood/cosX1cosX2_0_1_OOD', 'N_val.npy'))
dvf.show_2d_combinations([x_tr, x_te, x_val], features_names)