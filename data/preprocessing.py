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

def convertDatasetALGO(convertSample, editYALGO, X, Y, nb_annees = 5, window_size = 4, nb_decades = 37, progression_step = 20):
    new_X, new_Y = list(), list()
    last_progression = 0
    
    # Initialise the daily growths variables only
    Y_feats = editYALGO(Y, nb_decades, window_size)

    print(0, '%')
    for years_index in range(0, len(X) - nb_decades - window_size, nb_decades): # for each year
        # build samples by concatenating all the data in the window
        for window_index in range(years_index, years_index + nb_decades): # for each window
            
            x = convertSample(X, Y_feats, window_index, window_size)
            y = Y.iloc[window_index + window_size] # daily growth of the end of the window
    
            new_X.append(x)
            new_Y.append(y)
            
        progression = years_index / (len(X) - nb_decades) * 100
        if progression - last_progression > 1 and not int(progression) % progression_step:
            print(int(progression), '%')
            last_progression = progression
    print(100, '%')
    return new_X, new_Y


def editYSVR(Y, nb_decades, window_size, last_daily_growth= 7.17, first_daily_growths= 9.57):
    Y_feats = Y.copy()
    Y_feats.loc[Y_feats.index % nb_decades == (nb_decades-1)] = last_daily_growth # init of the last daily growth of each year
    Y_feats.loc[Y_feats.index % nb_decades < (window_size-1)] = first_daily_growths # init of the first daily growths of each year
    return Y_feats

def convertSampleSVR(X, Y, window_index, window_size):
    x_c = X.iloc[window_index:(window_index + window_size)] # concatenate climate data of entries that fit in the window
    x_g = Y.iloc[window_index:(window_index + window_size - 1)] # concatenate daily growth of the beginning of the window
    x_c = x_c.values.reshape(-1)
    x_g = x_g.values.reshape(-1)

    x = np.concatenate((x_c, x_g), axis = 0)
    return x

def convertDatasetSVR(X, Y, nb_annees = 5, window_size = 4, nb_decades = 37, progression_step = 20):
    return convertDatasetALGO(convertSampleSVR, editYSVR, X, Y, nb_annees, window_size, nb_decades, progression_step)


def editYLSTM(Y, nb_decades, window_size, last_daily_growth= 7.17, first_daily_growths= 9.57):
    Y_feats = Y.copy()
    Y_feats.loc[Y_feats.index % nb_decades == (nb_decades-1)] = last_daily_growth # init of the last daily growth of each year
    Y_feats.loc[Y_feats.index % nb_decades < (window_size-1)] = first_daily_growths # init of the first daily growths of each year
    Y_feats = pd.concat([pd.DataFrame([[last_daily_growth]], columns= Y_feats.columns), Y_feats])
    return Y_feats

def convertSampleLSTM(X, Y, window_index, window_size):
    x_c = X.iloc[window_index+1:(window_index + window_size)+1] # concatenate climate data of entries that fit in the window
    x_g = Y.iloc[window_index  :(window_index + window_size)] # concatenate daily growth of the beginning of the window
    x_g = x_g.values.reshape(window_size, 1)
    
    x = np.concatenate((x_c, x_g), axis = 1)
    return x

def convertDatasetLSTM(X, Y, nb_annees = 5, window_size = 4, nb_decades = 37, progression_step = 20):
    return convertDatasetALGO(convertSampleLSTM, editYLSTM, X, Y, nb_annees, window_size, nb_decades, progression_step)
