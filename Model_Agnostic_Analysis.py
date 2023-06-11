##########################################################################################
### Model_Agnostic_Analysis.py: Analyzing the trained model with partial dependence plots and feature importance ###
# Funding agency: Vlir-ous International Training Progamme (ITP) 2023 at Ghent university, Belgium
# Project title: Machine learning for climate change mitigation in buildings
# Author: Kim Q. Tran, CIRTech Institude, HUTECH university, Vietnam
# ! This work can be used, modified, and shared under the MIT License
# ! This work is included in the Github project: https://github.com/Kim-TranQuoc/Vlirous_ITP2023_UGent
# ! Email: tq.kim@hutech.edu.vn or tqkim.work@gmail.com
##########################################################################################

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
# from sklearn.inspection import partial_dependence
from pdpbox import pdp


def scale_data(data, data_range, scaled_range):
    scaled_data = data
    for i in range(data.shape[1]):
        scaler = MinMaxScaler(feature_range=scaled_range)
        scaler.fit(data_range[i, :].reshape(-1, 1))
        scaled_data[:, i] = scaler.transform(data[:, i].reshape(-1, 1)).flatten()
    return scaled_data


def inverse_scale_data(scaled_data, data_range, scaled_range):
    inv_scaled_data = scaled_data
    for i in range(scaled_data.shape[1]):
        scaler = MinMaxScaler(feature_range=scaled_range)
        scaler.fit(data_range[i, :].reshape(-1, 1))
        inv_scaled_data[:, i] = scaler.inverse_transform(scaled_data[:, i].reshape(-1, 1)).flatten()
    return inv_scaled_data


def plot_partial_dependence_plots(model, data_df, data_range, show=False, save_dir=None):
    pdp_values = {}
    for i in range(len(data_df.columns)):
        plot_feature_name = data_df.columns[i]
        pdp_feature = pdp.pdp_isolate(model=model, dataset=data_df, model_features=data_df.columns, feature=plot_feature_name)
        pdp_values[plot_feature_name] = pdp_feature.pdp
        fig, axes = pdp.pdp_plot(pdp_feature, plot_feature_name, plot_lines=True, frac_to_plot=0.5)
        fig.set_size_inches(8, 6)
        plt.title('Partial Dependence Plot of Feature ' + plot_feature_name, fontsize=14)
        plt.xlabel('Feature value', fontsize=10)
        plt.ylabel('Average change in predicted output', fontsize=10)
        ticks = data_df[plot_feature_name].unique()
        labels = inverse_scale_data(ticks.copy().reshape(-1, 1), np.asarray([data_range[i, :]]), (0, 1)).flatten()
        labels = ['{:.2f}'.format(value) for value in labels]
        plt.xticks(ticks=ticks, labels=labels)
        plt.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'PDP_' + plot_feature_name + '.png'), dpi=400)
    if show is not False:
        plt.show()
    return pdp_values


def plot_feature_importance(pdp_values, show=False, save_dir=None):
    print('----- Feature importance ----- ')
    feature_names = []
    feature_importances = []
    for feature, pdp_value in pdp_values.items():
        feature_names.append(feature)
        importance = np.sqrt(np.sum((pdp_value - np.mean(pdp_value))**2) / (len(pdp_value) - 1))
        feature_importances.append(importance)
        print(f'> {feature}: {importance:.3f}')
    fig = plt.figure(figsize=(6, 8))
    feature_names.reverse()
    feature_importances.reverse()
    bars = plt.barh(feature_names, feature_importances)
    for bar in bars:
        width = bar.get_width()
        if np.isnan(width):
            continue
        plt.text(width+0.2, bar.get_y() + bar.get_height() / 2, f'{width:.3f}', ha='left', va='center')
    plt.xlim(0, 25)
    plt.title('Feature Importance', fontsize=14)
    plt.xlabel('Importance factor', fontsize=10)
    plt.ylabel('Feature', fontsize=10)
    plt.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'Feature_Importances.png'), dpi=400)
    if show is not False:
        plt.show()
    return


def main():
    total_time_start = timeit.default_timer()

    # Make result directory
    saved_model = ['Model_3', '230610_005527']
    script_dir = os.path.dirname(__file__)
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    dt_string = datetime.now().strftime("%y%m%d_%H%M%S")
    result_dir = os.path.join(script_dir, 'Results', current_file, dt_string, saved_model[0] + '-' + saved_model[1])
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # Get new data
    data_range = np.array([[0, 5], [100, 500], [50, 350], [0.5, 1.5], [1, 3], [0.4, 1.6], [18, 28]])
    num_vars = np.array([6, 6, 7, 6, 3, 7, 6])
    feature_names = ['OTC', 'EWT', 'IWT', 'WBC', 'NGL', 'RTR', 'IAT']
    input_vars = {}
    for i in range(len(feature_names)):
        input_vars[feature_names[i]] = np.linspace(data_range[i, 0], data_range[i, 1], num_vars[i])
    values = np.array(list(input_vars.values()), dtype=object)
    new_data = np.array(np.meshgrid(*values, indexing='ij')).T.reshape(-1, len(values))
    new_data = scale_data(new_data, data_range, (0, 1))
    data_df = pd.DataFrame(new_data, columns=feature_names)

    # Load model
    saved_model = os.path.join(*saved_model)
    model_path = os.path.join(script_dir, 'Results', 'Train_Test', saved_model, 'Best_Model.h5')
    model = keras.models.load_model(model_path)
    # model.summary()

    # Partial dependence plots (PDP)
    pdp_values = plot_partial_dependence_plots(model, data_df, data_range, show=False, save_dir=result_dir)

    # Feature importance
    plot_feature_importance(pdp_values, show=False, save_dir=result_dir)  # Only after PDP

    # Total time
    total_time_stop = timeit.default_timer()
    print(f'>> Total running time = {total_time_stop - total_time_start:.3f}s')


if __name__ == '__main__':
    main()
