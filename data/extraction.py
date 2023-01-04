# Import libraries

import numpy as np
import pandas as pd

import pyreadr
import pickle
import os


# Import data

def importRawData(directory = "dataverse_files/"):
    data_dir_path = os.getcwd() + '/' + directory

    croissance_et_climat_decadaires = pyreadr.read_r(data_dir_path + "croissance_et_climat_decadaires.rds" ).popitem()[1]
    valorisation_annuelle = pyreadr.read_r(data_dir_path + "valorisation_annuelle.rds" ).popitem()[1]
    
    return croissance_et_climat_decadaires, valorisation_annuelle


# Data preparation

def prepareGrowthAndDecadalClimates(croissance_et_climat_decadaires, nb_annees):
    identifier_columns = ["ucs", "safran", "sol"] + ["type_de_prairie"] + ["gestion"]
    valuable_columns = identifier_columns + ["annee", "decade", "Tmin", "Tmax", "Tmoy", "Rain", "RG", "im", "croissance"]
    
    df = croissance_et_climat_decadaires.loc[:, valuable_columns]

    annee_inf = df["annee"].max() - nb_annees
    df = df.loc[(df["annee"] > annee_inf), :]

    df.sort_values(by= identifier_columns + ["annee", "decade"], inplace=True)

    df.dropna(inplace= True)

    df["annee"] = df["annee"].astype("int64")
    df["decade"] = df["decade"].astype("int64")

    return df


def prepareAnnualValuations(valorisation_annuelle):
    valuable_columns = ["ucs", "safran", "sol", "gestion"] + ["annee", "cumul_croissance"]
    df = valorisation_annuelle.loc[:, valuable_columns]

    df = df.reset_index(drop=True)
    df[["annee"]] = df[["annee"]].astype("int64")

    return df


def concatenateCcdVa(ccd_prepared, va_prepared):
    identifier_columns = ["ucs", "safran", "sol", "gestion"]
    dataset = ccd_prepared.merge(va_prepared, how='left', on= identifier_columns + ["annee"])
    
    # Cleansing of abominations
    identifier_columns[3:3] += ["type_de_prairie"]
    WrongCumul = dataset.groupby(identifier_columns+["annee"]).sum(numeric_only= True)["croissance"] / 100 != dataset.groupby(identifier_columns + ["annee"]).last()["cumul_croissance"]

    dataset = dataset.merge(WrongCumul.reset_index(), how='left', on= identifier_columns+["annee"])
    WrongIndexes = dataset[dataset[0] == True].index

    dataset.drop(WrongIndexes, inplace= True)
    dataset.drop(columns=0, inplace= True)
    
    return dataset


# Serialisation

def saveDataset(d, name):
    with open("datasets/"+name+".pickle", "wb") as outfile:
        pickle.dump(d, outfile)

def loadDataset(name):
    with open("datasets/"+name+".pickle", "rb") as infile:
        return pickle.load(infile)
