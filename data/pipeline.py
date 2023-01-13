# Import libraries

from data.extraction import *
from data.preprocessing import *

def build(model):
    # Import data
    croissance_et_climat_decadaires, valorisation_annuelle = importRawData("dataverse_files/")
    
    # Preprocessing
    nb_annees = 5
    ccd_prepared = prepareGrowthAndDecadalClimates(croissance_et_climat_decadaires, nb_annees)
    va_prepared = prepareAnnualValuations(valorisation_annuelle)
    dataset = concatenateCcdVa(ccd_prepared, va_prepared)
    
    if model.lower() == "svr":
        X_train, Y_train, X_test, Y_test = splitDatasetGeo(dataset)
        print("Converting training dataset for", model, "dataset")
        X_train, Y_train, Y_train_prior = convertDatasetSVR(X_train, Y_train, nb_annees)
        print()
        print("Converting testing dataset for", model, "dataset")
        X_test, Y_test, Y_test_prior = convertDatasetSVR(X_test, Y_test, nb_annees)
        print()
        print("Done!")
        return X_train, Y_train, Y_train_prior, X_test, Y_test, Y_test_prior
    elif model.lower() == "lstm":
        X_train, Y_train, X_val, Y_val, X_test, Y_test = splitDatasetGeo(dataset, validation= True)
        print("Converting training dataset for", model, "dataset")
        X_train, Y_train, Y_train_prior = convertDatasetLSTM(X_train, Y_train, nb_annees)
        print()
        print("Converting validation dataset for", model, "dataset")
        X_val, Y_val, _ = convertDatasetLSTM(X_val, Y_val, nb_annees)
        print()
        print("Converting testing dataset for", model, "dataset")
        X_test, Y_test, Y_test_prior = convertDatasetLSTM(X_test, Y_test, nb_annees)
        print()
        print("Done!")
        return X_train, Y_train, Y_train_prior, X_val, Y_val, X_test, Y_test, Y_test_prior
    else:
        print("No conversion after split")
    print()

    
def buildAndSave(model):
    print("Saving", model, "dataset")
    if model.lower() == "svr":
        X_train, Y_train, Y_train_prior, X_test, Y_test, Y_test_prior = build(model)
    if model.lower() == "lstm":
        X_train, Y_train, Y_train_prior, X_val, Y_val, X_test, Y_test, Y_test_prior = build(model)
        saveDataset((X_val, Y_val), model.lower()+"_val")
    saveDataset((X_train, Y_train, Y_train_prior), model.lower()+"_train")
    saveDataset((X_test, Y_test, Y_test_prior), model.lower()+"_test")
    print("Done!")
    if model.lower() == "svr":
        return X_train, Y_train, Y_train_prior, X_test, Y_test, Y_test_prior
    if model.lower() == "lstm":
        return X_train, Y_train, Y_train_prior, X_val, Y_val, X_test, Y_test, Y_test_prior


def load(model):
    print("Loading", model, "dataset")
    X_train, Y_train, Y_train_prior = loadDataset(model.lower()+"_train")
    X_test, Y_test, Y_test_prior = loadDataset(model.lower()+"_test")
    if model.lower() == "lstm":
        X_val, Y_val = loadDataset(model.lower()+"_val")
        print("Done!")
        return X_train, Y_train, Y_train_prior, X_val, Y_val, X_test, Y_test, Y_test_prior
    print("Done!")
    return X_train, Y_train, Y_train_prior, X_test, Y_test, Y_test_prior
