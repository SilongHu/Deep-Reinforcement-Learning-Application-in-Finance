from __future__ import absolute_import
import json
import datetime as dt
import pandas as pd
import numpy as np
from DRLApp.generate import load_config_data
from DRLApp.generate import load_log_data
import os
def generate_whole_set(choice):
    '''
    generate the training_set according to the Zhengyao's result
    '''

    result = load_config_data(choice)
    fake_train_end,train_end, coin_list = load_log_data(choice, result[3])

    dir = os.path.join("database",choice)
    list = os.listdir(dir)  # dir is your directory path
    number_files = len(list)/2

    newDict = {}
    scan_list = []
    raw_lower_df = pd.DataFrame()
    for i in range(number_files):
        raw_lower_df = pd.read_csv(os.path.join("database",choice,"lowerdata%s.csv" %i))
        raw_upper_df = pd.read_csv(os.path.join("database", choice, "upperdata%s.csv" % i))
        pp1 = raw_lower_df.keys()
        pp2 = raw_upper_df.keys()
        for item1,item2 in zip(pp1,pp2):
            if item1 == 'date' or item1 == 'isPartial':
                continue
            if item1 == 'bitcoin':
                divide = np.array(raw_lower_df[item1].tolist()) + np.array(raw_upper_df[item2].tolist())
                index = 0
                for each in divide:
                    # If the first element in bitcoin is 0
                    # Assign BTN trends average to it.
                    if each == 0 and index == 0:
                        divide[index] = np.mean(divide)
                    # If the bitcoin is 0, assign former value to it.
                    if each == 0 and index !=0:
                        divide[index] = divide[index - 1]
                    index += 1

                for each in newDict:
                    if each not in scan_list:
                        newDict[each] = newDict[each]/np.double(divide)
                        scan_list.append(each)
            else:
                newDict[item2] =np.array(raw_lower_df[item1].tolist())+ np.array(raw_upper_df[item2].tolist())

    with open(os.path.dirname(__file__) + "/../" + "train_package/" + str(choice) + "/net_config.json") as file:
        config = json.load(file)
    input_info = config['input']
    end_date = input_info['end_date']
    start_date = input_info['start_date']
    end_date = str(end_date).replace('/', '-', 2)
    start_date = str(start_date).replace('/', '-', 2)
    end_timedate = dt.datetime.strptime(end_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
    start_timedate = dt.datetime.strptime(start_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

    total_sample = (end_timedate - start_timedate).total_seconds()/input_info['global_period'] + 1
    ori_training_number = int(total_sample * (1 - input_info['test_portion'])) - input_info['window_size']-1
    ori_testing_number = int(total_sample - ori_training_number) - 2*input_info['window_size']-2

    real_training_number = ori_training_number/2 # index of end for training
    # start index of testing
    index_of_training_end = real_training_number + (input_info['window_size']+1)/2
    index_of_real_testing_end = index_of_training_end+ori_testing_number/2
    index_of_end = index_of_real_testing_end+(input_info['window_size']+1)/2

    return newDict,coin_list,index_of_training_end,index_of_real_testing_end,real_training_number

def generate_individual_set(coin_name,dict,real_training_number,input_size,num_steps,normalized):
    seq = [np.array(dict[coin_name][i * input_size: (i + 1) * input_size])
           for i in range(len(dict[coin_name]) // input_size)]
    # replace 0 in individual_set
    index = 0
    for each in seq:
        if each[0] == 0 and index == 0:
            each[0] = np.mean(seq)
        if each[0] == 0 and index != 0:
            each[0] = seq[index-1][0]
        index += 1

    if normalized:
        seq = [seq[0] / seq[0][0] ] + [
            curr / seq[i][-1]  for i, curr in enumerate(seq[1:])]

    X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])
    y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])
    train_X, test_X = X[:real_training_number], X[real_training_number:]
    train_y, test_y = y[:real_training_number], y[real_training_number:]
    return train_X, test_X, train_y, test_y