# Import libraries

from data_import import *
from data_preprocessing import *


def build(model):
    # Import data
    croissance_et_climat_decadaires, valorisation_annuelle = importRawData("dataverse_files/")
    
    # Preprocessing
    nb_annees = 5
    ccd_prepared = prepareGrowthAndDecadalClimates(croissance_et_climat_decadaires, nb_annees)
    va_prepared = prepareAnnualValuations(valorisation_annuelle)
    dataset = concatenateCcdVa(ccd_prepared, va_prepared)
    X_train, Y_train, X_test, Y_test = splitDatasetGeo(dataset)
    
    print("Converting training dataset for", model, "dataset")
    if model.lower() == "svr":
        X_train, Y_train = convertDatasetSVR(X_train, Y_train, nb_annees)
    if model.lower() == "lstm":
        X_train, Y_train = convertDatasetLSTM(X_train, Y_train, nb_annees)
    print()
    print("Converting testing dataset for", model, "dataset")
    if model.lower() == "svr":
        X_test, Y_test = convertDatasetSVR(X_test, Y_test, nb_annees)
    if model.lower() == "lstm":
        X_test, Y_test = convertDatasetLSTM(X_test, Y_test, nb_annees)
    print()
    print("Done!")
    
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

    
def buildAndSave(model):
    X_train, Y_train, X_test, Y_test = build(model)
    print()
    print("Saving", model, "dataset")
    saveDataset((X_train, Y_train), model.lower()+"_train")
    saveDataset((X_test, Y_test), model.lower()+"_test")
    print("Done!")
    return X_train, Y_train, X_test, Y_test


def load(model):
    print("Loading", model, "dataset")
    X_train, Y_train = loadDataset(model.lower()+"_train")
    X_test, Y_test = loadDataset(model.lower()+"_test")
    print("Done!")
    return X_train, Y_train, X_test, Y_test
