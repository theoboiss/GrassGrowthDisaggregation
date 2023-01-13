# AI Project - Polytech Tours

## Grass growth disaggregation: Study and development of a deep learning solution for plant growth prediction

Supervised by Prof. Dr. Nicolas RAGOT

Conducted by Seunkam EZRA and Théo BOISSEAU


## Description:

Implementation of an LSTM algorithm to disaggregate the cumulative annual grass growth by reconstructing the dynamics of the growth over the year using time series describing the climate of the year.

The disaggregation problem can also be seen as a special case of a more general problem: that of estimating individual values from aggregated data. This problem is also frequently encountered in economic applications where statistical data (for example, voting statistics) are known only for groups of people, but not individually.$^{(1)}$

(1) Temporal Disaggregation of the Cumulative Grass Growth, Thomas Guyet, Laurent Spillemaecker, Simon Malinowski, and Anne-Isabelle Graux, ICPRAI 2022, Paris.


## Project structure

data: Data preparation and pre-processing
- extraction.py: Extract raw or serialized data from files
- preprocessing.py: Transform structured data into model-specific datasets
- pipeline.py: Build all the data pipeline
- window.py: Define the sliding window to reinject LSTM's predictions in its features

lstm_model: Serialized lstm model
svr_model.pkl: Serialized svr model
svr_automl.pkl: Serialized svr tuner

datasets: Serialized datasets (if not created, you have to create it manually)
dataverse_files: Raw data (if not downloaded, you have to get it at https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.15454/FD9FHU)

data_analysis.ipynb: Study and explanation of all the data pipeline
svr.ipynb: Training and evaluation of SVR algorithm
lstm.ipynb: Training and evaluation of LSTM algorithm
SlidingWindow_tests.ipynb: Quality control of the sliding window for LSTM's reinjections
