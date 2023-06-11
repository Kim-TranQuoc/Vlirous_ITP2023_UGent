##########################################################################################
### Predict_Model.py: Generating predictions via a saved model ###
# Funding agency: Vlir-ous International Training Progamme (ITP) 2023 at Ghent university, Belgium
# Project title: Machine learning for climate change mitigation in buildings
# Author: Kim Q. Tran, CIRTech Institude, HUTECH university, Vietnam
# ! This work can be used, modified, and shared under the MIT License
# ! This work is included in the Github project: https://github.com/Kim-TranQuoc/Vlirous_ITP2023_UGent
# ! Email: tq.kim@hutech.edu.vn or tqkim.work@gmail.com
##########################################################################################

import tensorflow as tf
from tensorflow import keras
# from keras.layers.core import Activation
# from keras.utils.generic_utils import get_custom_objects
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
# from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import time
import os
from datetime import datetime
import psutil
from os import getpid
import shutil


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


def main():
    total_time_start = timeit.default_timer()

    # Make result directory
    saved_model = ['Model_3', '230609_211743']
    script_dir = os.path.dirname(__file__)
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    dt_string = datetime.now().strftime("%y%m%d_%H%M%S")
    result_dir = os.path.join(script_dir, 'Results', 'Predict', dt_string, saved_model[0] + '-' + saved_model[1])
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # Load saved model
    saved_model = os.path.join(*saved_model)
    model_path = os.path.join(script_dir, 'Results', 'Train_Test', saved_model, 'Best_Model.h5')
    shutil.copy2(model_path, result_dir)
    model = keras.models.load_model(model_path)

    # New data
    new_cases_path = os.path.join(script_dir, 'Results', 'Predict', 'New_cases')
    new_cases_files = [file for file in os.listdir(new_cases_path) if file.endswith(".csv")]
    for file_no, file_name in enumerate(new_cases_files):
        # New cases
        data_df = pd.read_csv(os.path.join(new_cases_path, file_name), header=None)
        new_data = data_df.to_numpy()
        case_name = file_name.replace('.csv', '')

        print('----- Data summary -----')
        print(f'> Data cases: {case_name} case')
        print(f'> Number of new data: {new_data.shape[0]} samples')

        # Scale data
        data_range = np.array([[0, 5], [100, 500], [50, 350], [0.5, 1.5], [1, 3], [0.4, 1.6], [18, 28]])
        scaled_range = (0, 1)
        new_data = scale_data(new_data, data_range, scaled_range)

        # Predict new data
        predict_train_y = model.predict(new_data)
        new_data = inverse_scale_data(new_data, data_range, scaled_range)
        new_df = pd.DataFrame(np.hstack((new_data, predict_train_y)))
        new_df.to_csv(os.path.join(result_dir, 'Predict_' + case_name + '.csv'), header=None, index=False)

    # input_vars = {'OTC': np.unique(np.append([1.3, 1.8, 3.4], [0 + i*0.5*2 for i in range(6)])),  # [0, 5]
    #               'EWT': np.unique(np.append([200, 300], [100 + i*40*2 for i in range(6)])),  # [0, 5]
    #               'IWT': np.unique(np.append([100, 200], [100 + i*20*2 for i in range(6)])),  # [0, 5]
    #               'WBC': np.unique(np.append([0.665, 0.9, 1.2], [0.5 + i*0.1*2 for i in range(6)])),  # [0, 5]
    #               'NGL': np.unique(np.append([1, 2, 3], [1 + i*1 for i in range(5)])),  # [1, 2, 3, 4, 5]
    #               'RTR': np.unique(np.append([0.6, 1.0, 1.4], [0.5 + i*0.2*2 for i in range(6)])),  # [0, 5]
    #               'IAT': np.unique(np.append([20, 22, 25], [18 + i*1*2 for i in range(6)]))  # [0, 5]
    #               }
    # values = np.array(list(input_vars.values()), dtype=object)
    # new_data = np.array(np.meshgrid(*values, indexing='ij')).T.reshape(-1, len(values))
    # print(f'> Number of new data: {new_data.shape[0]} samples')

    # predict_train_y = model.predict(new_data)
    # new_df = pd.DataFrame(np.hstack((new_data, predict_train_y)))
    # new_df.to_csv(os.path.join(result_dir, 'Predict_New_Data' + '.csv'), header=None, index=False)

    # Total time
    total_time_stop = timeit.default_timer()
    print(f'>> Total running time = {total_time_stop - total_time_start:.3f}s')


if __name__ == '__main__':
    main()
