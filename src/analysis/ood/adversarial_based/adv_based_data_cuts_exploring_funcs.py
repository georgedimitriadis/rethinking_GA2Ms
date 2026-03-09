
import os
import numpy as np
import pandas as pd
import pickle

ood_data_path = 'talent_benchmark/data_ood/adversarial_based/2026_01_28'
datasets = os.listdir(ood_data_path)
df_name = 'ood_results_df.pcl'


df = pd.DataFrame(columns=['dataset', 'epoch', 'repetition', 'ood_cut_r2'])

for dataset in datasets:
    dataset_df = pd.read_pickle(os.path.join(ood_data_path, dataset, df_name))
    temp_df = dataset_df[['repetition', 'epoch', 'ood_cut_r2']]
    name = [dataset for _ in range(len(temp_df))]
    temp_df = temp_df.assign(dataset = name)
    df = pd.concat([df, temp_df])

df = df.reset_index(drop=True)


# ADD TO THE DF THE TRAIN AND VAL INDICES FROM THE ood_cut_indices FILE AND RESAVE IT
for dataset in datasets:
    dataset_df = pd.read_pickle(os.path.join(ood_data_path, dataset, df_name))
    with open(os.path.join(ood_data_path, dataset, 'ood_cut_indices.pcl'), 'rb') as f:
        ood_cut_indices = pickle.load(f)

    if not 'train_indices' in dataset_df.columns:
        print(dataset)
        dataset_df['train_indices'] = np.nan
        dataset_df['val_indices'] = np.nan
        dataset_df['train_indices'] = dataset_df['train_indices'].astype(object)
        dataset_df['val_indices'] = dataset_df['val_indices'].astype(object)
        for i, (ep, rep) in enumerate(zip(dataset_df['epoch'], dataset_df['repetition'])):
            dataset_df.at[i, 'train_indices'] = ood_cut_indices[ep][rep]['train']
            dataset_df.at[i, 'val_indices'] = ood_cut_indices[ep][rep]['val']

        dataset_df.to_pickle(os.path.join(ood_data_path, dataset, df_name))



