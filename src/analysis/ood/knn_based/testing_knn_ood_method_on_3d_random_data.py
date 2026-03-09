
import numpy as np
from matplotlib import pyplot as plt

from src.analysis.utils import funcs_knn_ood_data_generation as ogkf
from src.analysis.visualisation import data_visualisation_funcs as dvf

num_of_features = 3
num_of_datapoints = 1000
features_ranges = [[0, 1], [-5, 10], [100, 1000]]

data = np.random.random((num_of_datapoints, num_of_features))
data[500] = [0.5, 0.5, 0.5]
for i in range(num_of_features):
    data[:, i] = ((data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min()))*(features_ranges[i][1] - features_ranges[i][0]) + features_ranges[i][0]

max_allowed_num_of_features_at_edge = 0

list_of_non_edge_datapoints_indices = ogkf.get_non_edge_point_indices(data,
                                                                     max_allowed_num_of_features_at_edge=max_allowed_num_of_features_at_edge)
array_of_non_edge_datapoints = data[list_of_non_edge_datapoints_indices]

for_point = np.argwhere(data[500] == array_of_non_edge_datapoints)[0][0]

all_possible_kneighbors_groups, _ = \
    ogkf.get_knn_based_point_groups(array_of_non_edge_datapoints, 0.1, [for_point])

p_indx=0
a_random_set_of_points = array_of_non_edge_datapoints[(list(all_possible_kneighbors_groups[2]))[p_indx], :]

dvf.show_2d_combinations([array_of_non_edge_datapoints, a_random_set_of_points], ['1', '2', '3'])

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2])
ax.scatter(xs=a_random_set_of_points[:, 0],
           ys=a_random_set_of_points[:, 1],
           zs=a_random_set_of_points[:, 2], c='r', s=100)
ax.scatter(xs=array_of_non_edge_datapoints[for_point, 0],
           ys=array_of_non_edge_datapoints[for_point, 1],
           zs=array_of_non_edge_datapoints[for_point, 2], c='g', s=300)


