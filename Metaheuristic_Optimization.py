##########################################################################################
### Metaheuristic_Optimization.py: Optimizing the problem with ANN surrogate model and metaheursitic algorithm DE ###
# Funding agency: Vlir-ous International Training Progamme (ITP) 2023 at Ghent university, Belgium
# Project title: Machine learning for climate change mitigation in buildings
# Author: Kim Q. Tran, CIRTech Institude, HUTECH university, Vietnam
# ! This work can be used, modified, and shared under the MIT License
# ! This work is included in the Github project: https://github.com/Kim-TranQuoc/Vlirous_ITP2023_UGent
# ! Email: tq.kim@hutech.edu.vn or tqkim.work@gmail.com
##########################################################################################

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import numpy.matlib
from random import random, sample
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


def Evaluate_Objective(x, DesVar_position, FixVar_val, FixVar_position, data_range, scaled_range, model, PopSize):
    cost_total = np.zeros((PopSize, 1), dtype=float)
    cost_initial = np.zeros((PopSize, 1), dtype=float)
    cost_operate = np.zeros((PopSize, 1), dtype=float)

    for i_ind in range(PopSize):
        # Scale variables
        FullVar = np.zeros((30, len(DesVar_position)+len(FixVar_position)))
        FullVar[:, DesVar_position] = np.matlib.repmat(x[i_ind, :], 30, 1)
        FullVar[:, FixVar_position] = FixVar_val
        FullVar_scale = scale_data(FullVar, data_range, scaled_range)

        # Predictions
        pred = model.predict(FullVar_scale)

        # Objective function
        (EWT, IWT, WBC, RTR) = (x[i_ind, 0]/1000, x[i_ind, 1]/1000, x[i_ind, 2], x[i_ind, 3])
        cost_exwall = 170588*(0.337*WBC**2 + 1.111*WBC)*EWT + 10820*EWT + 702
        cost_inwall = 55966*IWT + 242
        cost_roof = 331*RTR**(-2) + 4847*RTR**(-1) + 1585

        cost_initial[i_ind, 0] = cost_exwall + cost_inwall + cost_roof
        cost_operate[i_ind, 0] = np.sum(pred)*1000 * 9.13/100
        cost_total[i_ind, 0] = cost_initial[i_ind] + cost_operate[i_ind]
    return cost_total, cost_initial, cost_operate


def DE(DesVar_bound, DesVar_position, FixVar_val, FixVar_position, data_range, scaled_range, model, PopSize, MaxIter, tol, verbose=0):
    DesVar_num = DesVar_bound.shape[1]
    LB = DesVar_bound[0, :]
    UB = DesVar_bound[1, :]

    # Phase 1: Initialization
    # Target vector
    x = np.matlib.repmat(LB, PopSize, 1) + np.random.uniform(0, 1, (PopSize, DesVar_num)) * np.matlib.repmat(UB - LB, PopSize, 1)

    # Evaluate the objective function
    f, cost_initial, cost_operate = Evaluate_Objective(x, DesVar_position, FixVar_val, FixVar_position, data_range, scaled_range, model, PopSize)
    index = np.argmin(f, axis=0)
    fbest = f[index, 0].item()
    xbest = np.asarray(x[index, :])
    fmean = np.mean(f).item()
    delta = abs(abs(fmean)/abs(fbest) - 1)
    history = {'Iteration': [0],
               'xbest': xbest,
               'fbest': [fbest],
               'fmean': [fmean],
               'delta': [delta],
               'cost_initial': [cost_initial],
               'cost_operate': [cost_operate]
               }

    # Mutant vector
    v = np.zeros((PopSize, DesVar_num), dtype=float)
    # Trial vector
    u = np.zeros((PopSize, DesVar_num), dtype=float)

    for Iter in range(MaxIter):
        for i in range(PopSize):
            # Phase 2: Mutation
            # Randomly select R1, R2, R3 for rand/1 of DE
            candidates = list(range(0, PopSize))
            candidates.remove(i)
            R = np.asarray(sample(candidates, 3))

            F = 0.4 + (0.9 - 0.4)*np.random.uniform(0, 1)
            v_m = x[R[0], :] + F*(x[R[1], :] - x[R[2], :])

            # Return design variables violated bounds to the search space
            v_m = (v_m >= LB)*v_m + (v_m < LB)*(2*LB - v_m)
            v_m = (UB >= v_m)*v_m + (UB < v_m)*(2*UB - v_m)
            # v_m = (v_m >= LB)*v_m + (v_m < LB)*LB
            # v_m = (UB >= v_m)*v_m + (UB < v_m)*UB

            v[i, :] = v_m

            # Phase 3: Crossover
            K = sample(list(range(0, DesVar_num)), 1)
            Cr = 0.7 + (1 - 0.7)*np.random.uniform(0, 1)
            t = np.random.uniform(0, 1, DesVar_num) <= Cr
            t[K] = 1
            u[i, :] = t*v[i, :] + (1-t)*x[i, :]

        # Phase 4: Selection
        fnew, cost_initial_new, cost_operate_new = Evaluate_Objective(u, DesVar_position, FixVar_val, FixVar_position, data_range, scaled_range, model, PopSize)
        xnew = (fnew <= f)*u + (fnew > f)*x
        f = (fnew <= f)*fnew + (fnew > f)*f
        cost_initial = (fnew <= f)*cost_initial_new + (fnew > f)*cost_initial
        cost_operate = (fnew <= f)*cost_operate_new + (fnew > f)*cost_operate

        index = np.argmin(f, axis=0)
        xbest = np.asarray(xnew[index, :])
        fbest = f[index, 0].item()
        fmean = np.mean(f).item()
        cost_initial_best = cost_initial[index, 0].item()
        cost_operate_best = cost_operate[index, 0].item()

        # Prepare new iteration
        delta = abs(abs(fmean)/abs(fbest) - 1)
        x = xnew

        if verbose == 1:
            print(f'> Iteration {Iter:05}: fbest = {fbest:.3f}, fmean = {fmean:.3f}, delta_f = {delta:.5f}')
            print(f'                 : xbest = ', xbest)

        # Save history
        history['Iteration'].append(Iter)
        history['xbest'] = np.vstack((history['xbest'], xbest))
        history['fbest'].append(fbest)
        history['fmean'].append(fmean)
        history['delta'].append(delta)
        history['cost_initial'].append(cost_initial_best)
        history['cost_operate'].append(cost_operate_best)

        if delta <= tol or Iter == MaxIter:
            break

    return history


def main():
    total_time_start = timeit.default_timer()

    # Make result directory
    script_dir = os.path.dirname(__file__)
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    dt_string = datetime.now().strftime("%y%m%d_%H%M%S")
    result_dir = os.path.join(script_dir, 'Results', 'Optimization', current_file, dt_string)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # Get surrogate model
    saved_model = ['Model_3', '230610_005228']
    saved_model = str(saved_model[0]) + "-" + str(saved_model[1])
    model_path = os.path.join(script_dir, 'data', 'Surrogate_Models', saved_model, 'Best_Model.h5')
    shutil.copy2(model_path, result_dir)
    os.rename(os.path.join(result_dir, 'Best_Model.h5'), os.path.join(result_dir, saved_model + '.h5'))
    model_path = os.path.join(result_dir, saved_model + '.h5')
    model = keras.models.load_model(model_path)

    # Define optimization algorithm parameter
    Optimizer = 'DE'  # Differential evolution
    PopSize = 20
    MaxIter = 300
    tol = 1e-4
    RunTime = 5
    np.set_printoptions(formatter={'float': '{:.3f}'.format})

    # Define design variables boundaries
    DesVar_position = [1, 2, 3, 5]  # Start with 0
    DesVar_bound = np.array([[100, 50, 0.5, 0.4],
                             [500, 350, 1.5, 1.6]])

    # Define fixed variables of the surrogate model
    FixVar_position = [0, 4, 6]  # Start with 0
    FixVar_val = np.hstack((np.linspace(0, 1.8, 30).reshape(-1, 1), 2*np.ones((30, 1)), 28*np.ones((30, 1))))

    # Scale variables for the surrogate model
    data_range = np.array([[0, 5], [100, 500], [50, 350], [0.5, 1.5], [1, 3], [0.4, 1.6], [18, 28]])
    scaled_range = (0, 1)

    # Start task
    print('----- Processing -----')
    print(f'> Population size: {PopSize}')
    print(f'> Optimizer: {Optimizer}')
    print(f'> Total runtime: {RunTime}')
    Overall_report = []
    for i_run in range(RunTime):
        print(f'--- Run time = {i_run+1:03} --- ')

        optimizing_time_start = timeit.default_timer()
        history = DE(DesVar_bound, DesVar_position, FixVar_val, FixVar_position, data_range, scaled_range, model, PopSize, MaxIter, tol, verbose=1)
        optimizing_time_stop = timeit.default_timer()

        # #Display report
        (last_iter, fmean, delta) = (history['Iteration'][-1], history['fmean'][-1], history['delta'][-1])
        (fbest, xbest) = (history['fbest'][-1], history['xbest'][-1, :])
        (cost_initial, cost_operate) = (history['cost_initial'][-1], history['cost_operate'][-1])

        print(f'> Last iteration: {last_iter:05}')
        print(f'> Population result: - Mean total cost {fmean:.3f}')
        print(f'                     - Difference between best and mean total cost {delta:.5f}')
        print(f'> Best result: - Total cost: {fbest:.3f}')
        print(f'               - Initial cost: {cost_initial:.3f}')
        print(f'               - Operating cost: {cost_operate:.3f}')
        print(f'               - Solution:', xbest)
        print(f'> Training time = {optimizing_time_stop - optimizing_time_start:.3f}s')
        print('=====')
        print()

        # Save result
        Overall_report.append([PopSize, Optimizer, i_run+1, last_iter, fmean, fbest, delta, cost_initial, cost_operate,
                               xbest, optimizing_time_stop - optimizing_time_start])

        # Convergence history
        plt.figure('Convergence History')
        plt.plot(history['Iteration'], history['fbest'], c='b')
        plt.title('Optimization Convergence History')
        plt.xlabel('Iteration')
        plt.ylabel('Total cost')
        plt.savefig(os.path.join(result_dir, 'Convergence_History_of_Run_Number_' + str(i_run+1) + '.png'), dpi=400)
        # plt.show()
        plt.close('all')

    # Save result
    headers = ['Population size', 'Optimizer', 'Run time', 'Last iteration', 'Mean objective function',
               'Objective function (Total cost)', 'Delta objective function', 'Initial cost', 'Operating cost',
               'Solution', 'Optimizing Time (s)']

    new_df = pd.DataFrame(Overall_report)
    new_df.to_csv(os.path.join(result_dir, 'Result.csv'), header=headers, index=False)

    # Total time
    total_time_stop = timeit.default_timer()
    print(f'>> Total running time = {total_time_stop - total_time_start:.3f}s')


if __name__ == '__main__':
    # for i_opt in range(10):
    main()
