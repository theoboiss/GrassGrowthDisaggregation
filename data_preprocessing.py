# Import libraries

import numpy as np
import pandas as pd


# Data preprocessing

def splitDatasetGeo(dataset):
    safran_quantile_07 = dataset.drop_duplicates("safran")["safran"].quantile(0.7)
    train_index = dataset[dataset["safran"] <= safran_quantile_07].index
    test_index = dataset[dataset["safran"] > safran_quantile_07].index

    data_columns = ["Tmin", "Tmax", "Tmoy", "Rain", "RG", "im", "croissance", "cumul_croissance"]
    X = dataset.loc[:, data_columns]
    Y = pd.DataFrame({'croissance' : X.pop('croissance')})

    X_train = X[X.index.isin(train_index)]
    X_test = X[X.index.isin(test_index)]

    Y_train = Y[Y.index.isin(train_index)]
    Y_test = Y[Y.index.isin(test_index)]
    
    return X_train, Y_train, X_test, Y_test


# Adapt for algorithms

def convertDatasetSVR(X, Y, nb_annees = 5, window_size = 4, nb_decades = 37, progression_step = 20):
    new_X, new_Y = list(), list()
    last_progression = 0
    data_id_size = nb_decades*nb_annees # number of decades per year # PAS BATCH

    print(0, '%')
    for data_id_index in range(0, len(X) - data_id_size, data_id_size): # for each batch
        batch_end_index = data_id_index + data_id_size
        # build samples by concatenating all entries that fit in the window
        for index in range(data_id_index, batch_end_index - window_size): # for each window in the batch
            x_c = X.iloc[index:(index + window_size), :] # concatenate climate data of entries that fit in the window
            x_g = Y.iloc[index:(index + window_size - 1), 0] # concatenate daily growth of the beginning of the window
            x_c = x_c.values.reshape(-1)
            x_g = x_g.values.reshape(-1)
            x = list(x_c) + list(x_g)

            y = Y.iloc[index + window_size, 0] # daily growth of the end of the window
            new_X.append(x)
            new_Y.append(y)
            
        progression = data_id_index / (len(X) - data_id_size) * 100
        if progression - last_progression > 1 and not int(progression) % progression_step:
            print(int(progression), '%')
            last_progression = progression
    print(100, '%')
    return new_X, new_Y


def convertDatasetLSTM(X, Y, nb_annees = 5, window_size = 4, nb_decades = 37, progression_step = 20):
    new_X, new_Y = list(), list()
    last_progression = 0
    data_id_size = nb_decades*nb_annees # number of decades per year # PAS BATCH
    
    print(0, '%')
    for data_id_index in range(0, len(X) - data_id_size, data_id_size): # for each batch
        batch_end_index = data_id_index + data_id_size
        # build samples by concatenating all entries that fit in the window
        for index in range(data_id_index, batch_end_index - 1): # for each window in the batch # for each window in the batch
            #x_c = X.iloc[index+1] # concatenate climate data of entries that fit in the window
            #x_g = Y.iloc[index] # concatenate daily growth of the beginning of the window
            #x_c = x_c.values.reshape(-1)
            #x_g = x_g.values.reshape(-1)
            #x = list(x_c) + list(x_g)
            x = np.concatenate((X.iloc[[index+1]].values.reshape(-1), \
                                Y.iloc[[index]].values.reshape(-1)),axis = 0)
            y = Y.iloc[index+1] # daily growth of the end of the window
            new_X.append(x)
            new_Y.append(y)
            
        progression = data_id_index / (len(X) - data_id_size) * 100
        if progression - last_progression > 1 and not int(progression) % progression_step:
            print(int(progression), '%')
            last_progression = progression
    print(100, '%')
    return new_X, new_Y
