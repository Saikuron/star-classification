"""
Jean de Malliard
Class: CS 677
Date: 08/17/2022
Project
Description of Problem (just a 1-2 line summary!):
Find the best KNN
"""

from prepare_data import VERBOSE, X_train_encoded_scaled, \
                         X_test_encoded_scaled, Y_train, Y_test
from sklearn.neighbors import KNeighborsClassifier
from helper_functions import test_model
from sklearn.model_selection import GridSearchCV


def find_best_model():
    # Fit, predict, and compute accuracy/confusion matrix with the best
    # KNN model
    print(f"{' KNN ':=^100}")
    knn_model = KNeighborsClassifier()
    # Select the hyperparameters to test
    hyper_params = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid_search = GridSearchCV(estimator=knn_model,
                               param_grid=hyper_params,
                               n_jobs=-1, verbose=VERBOSE, scoring="accuracy")
    # Find the best KNN model using the different hyperparameters
    grid_search.fit(X_train_encoded_scaled, Y_train)
    # Print the best parameters
    print(f"Best KNN classifier hyperparameters: {grid_search.best_params_}")
    # Print the accuracy and confusion matrix
    test_model(grid_search.best_estimator_, X_train_encoded_scaled,
               X_test_encoded_scaled, Y_train, Y_test)
    print(f"{'=':=>100}", "\n")


if __name__ == "__main__":
    find_best_model()
