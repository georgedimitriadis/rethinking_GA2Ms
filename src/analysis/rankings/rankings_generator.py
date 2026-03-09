

import numpy as np
import pandas as pd
import choix

# Create rankings using Plackett-Luse and their plots

#src_directory = os.path.dirname(os.path.abspath(__file__))
reg_file_name = r'./talent_benchmark/results/regression.md'
bin_file_name = r'./talent_benchmark/results/cls_bin.md'
multi_file_name = r'./talent_benchmark/results/cls_multi.md'

def generate_df_from_md_data(md_file_path):
    with open(md_file_path, 'r') as f:
        reg_data_md = f.read()
    lines = reg_data_md.split("\n")
    header = lines[0].strip("|").split("|")
    data = []
    # Loop through lines starting from 2
    for line in lines[2:]:
        # Break once we hit an empty line
        if not line.strip():
            break
        cols = line.strip("|").split("|")
        row = dict(zip(header, cols))
        data.append(row)
    df = pd.DataFrame(data)

    datasets = df[' Dataset ']
    df_num = []
    for dataset in datasets:
        temp_dataset = df.iloc[np.where(df[' Dataset '] == dataset)[0][0], :][1:].to_list()
        temp_dataset = [float(a.split(' ')[1].split('+')[0].split('*')[-1]) for a in temp_dataset]
        df_num.append(temp_dataset)
    df_num = pd.DataFrame(df_num)
    df_num.columns = [a.split(' ')[1] for a in df.columns[1:]]
    df_num.index = [a.split(' ')[1] for a in df[' Dataset ']]
    return df_num

def generate_ordered_indices_for_ranking(df, using_accuracy=False):
    orders = []
    datasets = list(df.index)
    for dataset in datasets:
        model_rmse_in_dataset = df.loc[dataset].to_list()
        sorted_indices = np.argsort(model_rmse_in_dataset)[::-1] if using_accuracy else np.argsort(model_rmse_in_dataset)
        orders.append(sorted_indices)
    return orders

def generate_ordered_rankings_Plackett_Luse(md_file_name=None, df=None):
    if md_file_name is None:
        assert df is not None, print('Please provide either a .md file or a dataframe.')
        using_accuracy = False if 'boston' in df.index else True
    else:
        df = generate_df_from_md_data(md_file_name)
        using_accuracy = False if 'reg' in md_file_name else True

    models = df.columns
    orders_for_ranking = generate_ordered_indices_for_ranking(df, using_accuracy)

    ranking = choix.lsr_rankings(n_items=len(orders_for_ranking[0]), data=orders_for_ranking)
    tuples = sorted(zip(ranking, models))
    ranking_list, models_ranked = [t[0] for t in tuples], [t[1] for t in tuples]

    return ranking_list, models_ranked


def get_all_PL_rankings_from_md_files(reg_file_name, bin_file_name, multi_file_name):
    reg_ranking_list, reg_models_ranked = generate_ordered_rankings_Plackett_Luse(md_file_name=reg_file_name)
    bin_ranking_list, bin_models_ranked = generate_ordered_rankings_Plackett_Luse(md_file_name=bin_file_name)
    multi_ranking_list, multi_models_ranked = generate_ordered_rankings_Plackett_Luse(md_file_name=multi_file_name)

    return reg_ranking_list, reg_models_ranked, bin_ranking_list, bin_models_ranked, multi_models_ranked, multi_ranking_list


def get_all_PL_rankings_from_dfs(reg_df, bin_df, multi_df):
    reg_ranking_list, reg_models_ranked = generate_ordered_rankings_Plackett_Luse(df=reg_df)
    bin_ranking_list, bin_models_ranked = generate_ordered_rankings_Plackett_Luse(df=bin_df)
    multi_ranking_list, multi_models_ranked = generate_ordered_rankings_Plackett_Luse(df=multi_df)

    return reg_ranking_list, reg_models_ranked, bin_ranking_list, bin_models_ranked, multi_models_ranked, multi_ranking_list


# Create Rankings using difference to MLP
def create_differences_to_mlp(md_file_name=None, df=None):
    if md_file_name is None:
        assert df is not None, print('Please provide either a .md file or a dataframe.')
        multiplier = -1 if 'boston' in df.index  else 1
    else:
        df = generate_df_from_md_data(md_file_name)
        multiplier = -1 if 'reg' in md_file_name else 1

    df_diff = multiplier * df.subtract(df['mlp'], axis=0).divide(df['mlp'], axis=0)
    df_diff[np.abs(df_diff) > 10] = np.nan
    df_diff.index.name = 'Datasets'
    df_diff.columns.name = 'Models'

    return df_diff

def make_all_diffs_to_mlp_from_md_files(reg_file_name, bin_file_name, multi_file_name):
    reg_df_diff = create_differences_to_mlp(md_file_name=reg_file_name)
    bin_df_diff = create_differences_to_mlp(md_file_name=bin_file_name)
    multi_df_diff = create_differences_to_mlp(md_file_name=multi_file_name)

    return reg_df_diff, bin_df_diff, multi_df_diff

def make_all_diffs_to_mlp_from_dfs(df_reg, df_bin, df_multi):
    reg_df_diff = create_differences_to_mlp(df=df_reg)
    bin_df_diff = create_differences_to_mlp(df=df_bin)
    multi_df_diff = create_differences_to_mlp(df=df_multi)

    return reg_df_diff, bin_df_diff, multi_df_diff
