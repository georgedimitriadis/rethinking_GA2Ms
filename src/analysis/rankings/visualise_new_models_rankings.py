
import os
from matplotlib import pyplot as plt
import seaborn as sns
import analysis.rankings.rankings_generator as rp
import analysis.rankings.new_models_results as nmr

data_dir = './talent_benchmark/data'
datasets = os.listdir(data_dir)
logs_dir = './talent_benchmark/logs/2026_02_17_base'
the_log_dir = 'ecmac'
#logs_dir = os.path.join(logs_dir, the_log_dir)

def create_ranking_plot(ordered_model_names, ordered_rankings, type='Regression'):
    fig, ax = plt.subplots()
    ax.bar(ordered_model_names, ordered_rankings)
    ax.set_xticklabels(ordered_model_names, rotation=90, ha='right')
    ax.set_xlabel("Model")
    ax.set_ylabel("Plackett-Luce Ranking")
    ax.set_title(type)

def plot_all_PL_rankings(reg_ranking_list, reg_models_ranked, bin_ranking_list, bin_models_ranked, multi_models_ranked, multi_ranking_list):
    create_ranking_plot(reg_models_ranked, reg_ranking_list, type='Regression')
    create_ranking_plot(bin_models_ranked, bin_ranking_list, type='Binary Classification')
    create_ranking_plot(multi_models_ranked, multi_ranking_list, type='Full Classification')

def boxplot_diffs_to_mlp(df, type='Regression'):
    _ = plt.figure()
    a = sns.boxplot(df, orient='h')
    a.vlines(x=0, ymin=-1, ymax=len(df.columns))
    a.set_xlabel('Ratio better than MLP')
    a.set_ylabel('Models')
    a.set_title(type)

def make_all_boxplots(reg_df_diff, bin_df_diff, multi_df_diff):
    boxplot_diffs_to_mlp(reg_df_diff, 'Regression')
    boxplot_diffs_to_mlp(bin_df_diff, 'Binary Classification')
    boxplot_diffs_to_mlp(multi_df_diff, 'Full Classification')

df_reg, df_bin, df_multi = nmr.fill_dfs_with_model_results(logs_dir=logs_dir, datasets=datasets, model='ecmac')

reg_ranking_list, reg_models_ranked, bin_ranking_list, bin_models_ranked, multi_models_ranked, multi_ranking_list =\
    rp.get_all_PL_rankings_from_dfs(df_reg, df_bin, df_multi)
plot_all_PL_rankings(reg_ranking_list, reg_models_ranked, bin_ranking_list, bin_models_ranked, multi_models_ranked, multi_ranking_list)


reg_df_diff, bin_df_diff, multi_df_diff = rp.make_all_diffs_to_mlp_from_dfs(df_reg, df_bin, df_multi)
make_all_boxplots(reg_df_diff, bin_df_diff, multi_df_diff)
