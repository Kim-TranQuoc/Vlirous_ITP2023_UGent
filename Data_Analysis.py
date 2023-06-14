##########################################################################################
### Data_Analysis.py: Analyzing the simulation data with statistical features, histogram & density plots, data
# visualization with parallel coordinates plots, and variable correlation coefficients ###
# Funding agency: Vlir-ous International Training Progamme (ITP) 2023 at Ghent university, Belgium
# Project title: Machine learning for climate change mitigation in buildings
# Author: Kim Q. Tran, CIRTech Institude, HUTECH university, Vietnam
# ! This work can be used, modified, and shared under the MIT License
# ! This work is included in the Github project: https://github.com/Kim-TranQuoc/Vlirous_ITP2023_UGent
# ! Email: tq.kim@hutech.edu.vn or tqkim.work@gmail.com
##########################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import time
import os
from datetime import datetime
import psutil
from os import getpid
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px


def get_data():
    data_df = pd.read_csv('./data/Data_Infor.csv', header=None)
    data_infor = data_df.to_numpy()

    data_folder_path = './data/BaseVar'
    data_files = [file for file in os.listdir(data_folder_path) if file.endswith(".csv")]

    data_X = np.zeros([len(data_files), 7])
    data_y = np.zeros([len(data_files), 1])
    for file_no, file_name in enumerate(data_files):
        index = np.where(data_infor[:, 0] == file_name.replace('.csv', ''))
        data_X[file_no, :] = data_infor[index, 1:8].flatten()
        data_df = pd.read_csv(os.path.join(data_folder_path, file_name), header=None)
        data_arr = data_df.to_numpy()
        data_y[file_no, :] = np.sum(np.asarray(data_arr[0:3, :]))/1000  # From kWh to MWh

    print('----- Data summary -----')
    print(f'> Number of data samples: {len(data_files)} samples')
    return data_X, data_y


def plot_overall_histogram(data_df, show=False, save_dir=None):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Overall Histogram", fontsize=14)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    ax = fig.add_subplot(1, 1, 1)
    data_df.hist(ax=ax, bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
                 xlabelsize=10, ylabelsize=10, grid=False)
    plt.tight_layout(rect=(0, 0, 1, 1))
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'Overall_Histogram.png'), dpi=400)
    if show is not False:
        plt.show()
    return


def plot_histogram_and_density(data_df, nrows, ncols, show=False, save_dir=None):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    fig.suptitle("Histogram and Density Plots", fontsize=14)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    for i, col in enumerate(data_df.columns):
        ax1 = axes[i % nrows, (i//nrows) % ncols]
        ax1.set_title(col)
        ax1.set_xlabel("Value", fontsize=10)
        ax1.set_ylabel("Frequency", fontsize=10)
        ax1.hist(data_df[col], bins=15, color='steelblue', edgecolor='black', linewidth=1.0)
        ax2 = ax1.twinx()
        sns.kdeplot(data_df[col], ax=ax2, fill=False, color='orange', linewidth=1.5)
        ax2.set_ylabel("Density", fontsize=10)
    plt.grid(False)
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.subplots_adjust(hspace=0.7)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'Histogram_and_Density.png'), dpi=400)
    if show is not False:
        plt.show()
    return


def plot_parallel_coordinates(data_df, show=False, save_dir=None, color_var=0):
    fig = px.parallel_coordinates(data_df, color=color_var,
                                  color_continuous_scale=[[0, 'blue'], [1, 'red']], range_color=[30, 90], color_continuous_midpoint=2)
    fig.update_layout(title={'text': 'Parallel Coordinate Plot', 'x': 0.5, 'y': 1.0})
    fig.update_layout(font=dict(size=16))
    if save_dir is not None:
        fig.write_image(os.path.join(save_dir, 'Parallel_Coordinates.png'), format="png", width=1200, height=800, scale=2)
    if show is not False:
        fig.show()
    return fig


def plot_correlation_matrix(data_df, show=False, save_dir=None):
    corr_matrix_full = np.corrcoef(data_df, rowvar=False)
    num_var = corr_matrix_full.shape[0]
    new_df = pd.DataFrame(corr_matrix_full)
    new_df.to_csv(os.path.join(save_dir, 'Correlation_Matrix.csv'), header=None, index=False)
    plt.figure('Correlation Matrix')
    sns.heatmap(corr_matrix_full, annot=True, fmt=".3f", cmap="coolwarm", square=True)
    plt.xticks([i+0.5 for i in range(num_var)], ["Var." + str(i+1) for i in range(num_var)], rotation=45)
    plt.yticks([i+0.5 for i in range(num_var)], ["Var." + str(i+1) for i in range(num_var)], rotation=45)
    plt.title("Correlation Matrix Plot")
    plt.xlabel("Variables")
    plt.ylabel("Variables")
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'Correlation_Matrix.png'), dpi=400)
    if show is not False:
        plt.show()
    return


def main():
    total_time_start = timeit.default_timer()

    # Make result directory
    script_dir = os.path.dirname(__file__)
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    dt_string = datetime.now().strftime("%y%m%d_%H%M%S")
    result_dir = os.path.join(script_dir, 'Results', current_file, dt_string)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # Get data
    data_X, data_y = get_data()
    data = np.hstack((data_X, data_y))
    data_df = pd.DataFrame(data)
    new_names = ["OTC", "EWT", "IWT", "WBC", "NGL", "RTR", "IAT", "YER"]
    col_names = {col: new_names[i] for i, col in enumerate(data_df.columns)}
    data_df.rename(columns=col_names, inplace=True)
    data_df.to_csv(os.path.join(result_dir, 'Full_Data.csv'), header=None, index=False)

    # Statistical features
    new_df = pd.DataFrame(round(data_df.describe(), 2))
    new_df.to_csv(os.path.join(result_dir, 'Statistical_Features.csv'), index=True)

    # Histogram and Density plots
    # plot_overall_histogram(data_df, show=False, save_dir=result_dir)
    plot_histogram_and_density(data_df, nrows=4, ncols=2, show=False, save_dir=result_dir)

    # Visualize data with parallel coordinates plot
    plot_parallel_coordinates(data_df, show=False, save_dir=result_dir, color_var="YER")

    # Full variables correlation
    plot_correlation_matrix(data, show=False, save_dir=result_dir)

    # Total time
    total_time_stop = timeit.default_timer()
    print(f'>> Total running time = {total_time_stop - total_time_start:.3f}s')


if __name__ == '__main__':
    main()
    # get_data()
