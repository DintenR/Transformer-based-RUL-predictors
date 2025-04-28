# Transformer-based-RUL-predictors
This repository contains the models and scripts implemented to carry out the experiments described in the paper "Building of transformer-based RUL predictors supported
by explainability techniques: application on real industrial datasets". The repository is divided in three sections each one for each of the case studies: CMAPSS, MetroPT and Scania Component X.

In each directory are included:
 - The Pytorch Dataset class that performs all the data preparation
 - The different model architectures used for the case study
 - Scripts to train and test de models

requirements.txt contains the python dependecies that are needed to execute the scripts.

## To execute the experiments

1. Create a python environment
2. Install the requirements
3. Open a command prompt
4. Move to the case study directory and set the PYTHONPATH variable
5. To train a model run the *_training.py script.
6. To test a model run the test_model.py script.

## Datasets

The datasets used can be found and downloaded through the following URLs:
- [C-MAPSS](https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data)
- [MetroPT](https://zenodo.org/records/6854240) 
- [MetroPT2](https://zenodo.org/records/7766691) 
- [MetroPT3](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) 
- [Scania Component X](https://researchdata.se/en/catalogue/dataset/2024-34) 