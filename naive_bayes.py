"""
Jean de Malliard
Class: CS 677
Date: 08/18/2022
Project
Description of Problem (just a 1-2 line summary!):
Find the best Naive Bayes model
"""

from prepare_data import VERBOSE, X_train_encoded, \
                         X_test_encoded, Y_train, Y_test
from sklearn.naive_bayes import GaussianNB
from helper_functions import test_model
from sklearn.model_selection import GridSearchCV
import numpy as np


def find_best_model():
    # Fit, predict, and compute accuracy/confusion matrix with the best
    # Naive Bayes model
    print(f"{' Naive Bayes ':=^100}")
    nb_model = GaussianNB()
    # Select the hyperparameters to test
    hyper_params = {
        'var_smoothing': np.logspace(0, -9, num=100),
    }
    grid_search = GridSearchCV(estimator=nb_model,
                               param_grid=hyper_params,
                               n_jobs=-1, verbose=VERBOSE, scoring="accuracy")
    # Find the best KNN model using the different hyperparameters
    grid_search.fit(X_train_encoded, Y_train)
    # Print the best parameters
    print(f"Best NB classifier hyperparameters: {grid_search.best_params_}")
    # Print the accuracy and confusion matrix
    test_model(grid_search.best_estimator_, X_train_encoded, X_test_encoded,
               Y_train, Y_test)
    print(f"{'=':=>100}", "\n")


if __name__ == "__main__":
    find_best_model()
