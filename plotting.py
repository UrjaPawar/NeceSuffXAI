import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
name_dict = {"dist": "perturb",
"dist_random_opposite": "skewed(opp)",
"dist_random_same": "skewed(same)",
"dist_random_balanced": "balanced",
"dist_opposite": "perturb_skewed(opp)",
"dist_same": "perturb_skewed(same)",
"dist_balanced": "perturb_balanced",
"dist_outside": "perturb_outside",
"dist_inside": "perturb_inside",
"dist_random": "standard",
"dist_random_outside": "outside",
"dist_random_inside": "inside",

"generic": "generic_perturb",
"generic_random_opposite": "generic_skewed(opp)",
"generic_random_same": "generic_skewed(same)",
"generic_random_balanced": "generic_balanced",
"generic_opposite": "generic_perturb_skewed(opp)",
"generic_same": "generic_perturb_skewed(same)",
"generic_balanced": "generic_perturb_balanced",
"generic_outside": "generic_perturb_outside",
"generic_inside": "generic_perturb_inside",
"generic_random": "generic",
"generic_random_outside": "generic_outside",
"generic_random_inside": "generic_inside",

"standard": "standard",
"standard_outside": "standard_outside"
}

# Function to filter data based on selected configurations and methods
def filter_data(df, selected_config, selected_method):
    return df[(df['config'] == selected_config) & (df['methods'] == selected_method)]

# Function to prepare data for plotting
def prepare_plot_data(data, feature_length):
    grouped = data.groupby('data')['corr'].mean().reset_index()
    grouped['feature_length'] = feature_length
    return grouped

# Function to plot scatter data
def plot_scatter_with_mapping(data, color, label, marker_dict, data_map):
    for marker, group in data.groupby('data'):
        plt.scatter(group['feature_length'], group['corr'], color=color, s=100, label=f"{label} ({data_map.get(marker, marker)})", marker=marker_dict.get(marker, 'x'))
        plt.plot(group['feature_length'], group['corr'], linestyle='--', linewidth=2, color=color)

# Function to generate the main plot
def generate_main_plot(shap_data, lime_data, cf_data, title):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16)
    plot_scatter_with_mapping(shap_data, 'limegreen', 'SHAP-NECE', marker_styles, data_map)
    plot_scatter_with_mapping(lime_data, 'steelblue', 'LIME-NECE', marker_styles, data_map)
    plot_scatter_with_mapping(cf_data, 'coral', 'DiCE-NECE', marker_styles, data_map)

    plt.xlabel('Feature Subset Length',fontsize=16)
    plt.ylabel('Average Correlation',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xticks([1, 2, 3, 4])
    # plt.ylim(0, 1)
    plt.ylim(-0.6, 1)
    plt.tight_layout()
    plt.show()

# Function to generate the legend plot
def generate_legend_plot(unique_datasets):
    legend_labels = []
    for method, color, label in zip(['SHAP', 'LIME', 'DiCE'], ['limegreen', 'steelblue', 'coral'], ['SHAP', 'LIME', 'DiCE']):
        for dataset, marker in marker_styles.items():
            legend_labels.append(plt.Line2D([0], [0], color=color, marker=marker, linestyle='None',
                                            markersize=10, label=f"{label} ({data_map[dataset]})"))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.legend(handles=legend_labels, title="Legend", ncol=3, fontsize=9)
    ax.axis('off')
    plt.show()

# Function to generate the scatter plot and legend for multiple feature lengths
def generate_plots_for_feature_lengths(data_files, feature_lengths, ns, configs, methods, title):
    shap_plot_data = pd.DataFrame()
    lime_plot_data = pd.DataFrame()
    cf_plot_data = pd.DataFrame()

    for feature_length, data_file in zip(feature_lengths, data_files):
        df = pd.read_csv(data_file)
        if "NECE" in ns:
          shap_data_filtered = filter_data(df, configs['SHAP-NECE'], methods['SHAP-NECE'])
          lime_data_filtered = filter_data(df, configs['LIME-NECE'], methods['LIME-NECE'])
          cf_data_filtered = filter_data(df, configs['DiCE-NECE'], methods['DiCE-NECE'])
        if "SUFF" in ns:
          shap_data_filtered = filter_data(df, configs['SHAP-SUFF'], methods['SHAP-SUFF'])
          lime_data_filtered = filter_data(df, configs['LIME-SUFF'], methods['LIME-SUFF'])
          cf_data_filtered = filter_data(df, configs['DiCE-SUFF'], methods['DiCE-SUFF'])

        shap_plot_data = pd.concat([shap_plot_data, prepare_plot_data(shap_data_filtered, feature_length)])
        lime_plot_data = pd.concat([lime_plot_data, prepare_plot_data(lime_data_filtered, feature_length)])
        cf_plot_data = pd.concat([cf_plot_data, prepare_plot_data(cf_data_filtered, feature_length)])

    generate_main_plot(shap_plot_data, lime_plot_data, cf_plot_data, title)
    unique_datasets = pd.concat([shap_plot_data['data'], lime_plot_data['data'], cf_plot_data['data']]).unique()
    generate_legend_plot(unique_datasets)
data_files_nece = ['/content/gdrive/My Drive/xai_proj/conf/corrs_use.csv',
              '/content/gdrive/My Drive/xai_proj/conf/correlations_seq_2_final_nece.csv',
              '/content/gdrive/My Drive/xai_proj/conf/correlations_seq_3_final_nece.csv',
              '/content/gdrive/My Drive/xai_proj/conf/correlations_seq_4_final_nece.csv']
data_files_suff = ['/content/gdrive/My Drive/xai_proj/conf/corrs_use.csv',
              '/content/gdrive/My Drive/xai_proj/conf/correlations_seq_2_final_suff.csv',
              '/content/gdrive/My Drive/xai_proj/conf/correlations_seq_3_final_suff.csv',
              '/content/gdrive/My Drive/xai_proj/conf/correlations_seq_4_final_suff.csv']
# Configurations and methods to be used
# configs = {'SHAP-NECE': 'dist_outside', 'LIME-NECE': 'dist_balanced', 'DiCE-NECE': 'dist_outside',
#            'SHAP-SUFF': 'dist_outside', 'LIME-SUFF': 'dist_balanced', 'DiCE-SUFF': 'dist_outside'}
methods = {'SHAP-NECE': 'SHAP-NECE', 'LIME-NECE': 'LIME_weights-NECE', 'DiCE-NECE': 'CF-NECE',
          'SHAP-SUFF': 'SHAP-SUFF', 'LIME-SUFF': 'LIME_weights-SUFF', 'DiCE-SUFF': 'CF-SUFF'}
configs = {'SHAP-NECE': 'standard', 'LIME-NECE': 'standard', 'DiCE-NECE': 'standard_outside',
           'SHAP-SUFF': 'standard', 'LIME-SUFF': 'standard', 'DiCE-SUFF': 'standard_outside'}

data_map = {"heart":"Heart Disease",
              "diab":"Diabetes",
              "cerv":"Cervical Cancer"}
feature_lengths_updated = [1, 2, 3, 4]
marker_styles = {'cerv': 'o', 'heart': 's', 'diab': 'x'}



# Generate the plots
generate_plots_for_feature_lengths(data_files_suff, feature_lengths_updated,"SUFF",configs, methods, 'Average Correlations of XAI Frameworks with Sufficiency for Ranked Feature Sets')

# Update the function to allow specification of method names and to rename CF-SUFF as DiCE-SUFF
cors_df = pd.read_csv('/content/gdrive/My Drive/xai_proj/conf/corrs_use.csv')



# Map for method renaming
method_name_map = {
    'LIME_weights-SUFF': 'LIME-SUFF',
    'CF-SUFF': 'DiCE-SUFF'
}

df = pd.read_csv('corrs_use.csv')

def plot_method_corr(df, configs, x_order, method, a,b, x,y,color='mediumaquamarine'):
    # Filter the DataFrame to include rows related to the specified method
    filtered_df = df[df['methods'].str.contains(method)]

    # Filter to include only the specified configurations
    filtered_df = filtered_df[filtered_df['config'].isin(configs)]

    # Identify unique datasets
    unique_datasets = filtered_df['data'].unique()
    methods_arr = method.split("-")
    if methods_arr[0]=="LIME_weights": methods_arr[0]="LIME"
    elif methods_arr[0]=="CF": methods_arr[0]="DiCE"
    if methods_arr[1]=="SUFF": other = "Sufficiency"
    else:other = "Necessity"
    # Generate separate box plots for each dataset
    for dataset in unique_datasets:
        # plt.figure(figsize=(12, 6))
        plt.figure(figsize=(x,y))
        # plt.figure(figsize=(10, 5))
        sns.boxplot(x='config', y='corr', data=filtered_df[filtered_df['data'] == dataset], order=x_order, color=color, width = 0.5)
        plt.title(f'Correlations ({methods_arr[0]} - {other}) in {data_map[dataset]} Dataset', fontsize=16)
        plt.xticks(ticks=list(range(len(configs))), labels=[name_dict[label] for label in x_order], rotation=45)
        plt.xlabel('Neighborhoods', fontsize=17)
        plt.ylabel('Correlation (τ)', fontsize=17)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.ylim(a,b)
        plt.show()

# Test the function with SHAP-SUFF method

configs_shap = ['dist','dist_random_opposite', 'dist_random_same', 'dist_random_balanced', 'dist_opposite', 'dist_same',
 'dist_balanced', 'dist_outside', 'dist_inside', 'dist_random_outside', 'dist_random_inside', 'standard']
# configs_shap_order = ['perturb',  'inside', 'outside','balanced','skewed(opp)', 'skewed(same)', 'perturb_inside', 'perturb_outside', 'perturb_balanced',
#                       'perturb_skewed(opp)','perturb_skewed(same)', 'standard']
configs_shap_order=['dist', 'dist_random_inside','dist_random_outside','dist_random_balanced', 'dist_random_opposite',
                       'dist_random_same','dist_inside', 'dist_outside', 'dist_balanced','dist_opposite', 'dist_same',  'standard']

configs_lime = ['dist', 'dist_random_opposite', 'dist_random_same', 'dist_random_balanced', 'dist_opposite', 'dist_same',
 'dist_balanced', 'standard']
# configs_lime_order = ['perturb', 'balanced', 'skewed(opp)', 'skewed(same)','perturb_balanced', 'perturb_skewed(opp)', 'perturb_skewed(same)', 'standard']
configs_lime_order = ['dist', 'dist_random_balanced', 'dist_random_opposite', 'dist_random_same','dist_balanced', 'dist_opposite',  'dist_same', 'standard']

configs_cf = ['dist_outside', 'dist_random_outside', 'standard_outside']
configs_cf_order = ['dist_random_outside', 'dist_outside', 'standard_outside']
data_map = {"heart":"Heart Disease",
              "diab":"Diabetes",
              "cerv":"Cervical Cancer"}

# plot_method_corr(df, configs_lime, configs_lime_order,'LIME_weights-NECE',-0.1,1,14,7)
# plot_method_corr(df, configs_lime, configs_lime_order,'LIME_weights-SUFF',-0.1,1,14,7)
# plot_method_corr(df, configs_shap, configs_shap_order,'SHAP-NECE',-0.1,1.1,17,8)
# plot_method_corr(df, configs_shap, configs_shap_order,'SHAP-SUFF',-0.1,1.1,17,8)
# plot_method_corr(df, configs_cf, configs_cf_order,'CF-NECE',-0.3,1.1,8,5)
# plot_method_corr(df, configs_cf, configs_cf_order, 'CF-SUFF',-0.3,1.1,8,5)
