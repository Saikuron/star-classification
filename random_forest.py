"""
Jean de Malliard
Class: CS 677
Date: 08/17/2022
Project
Description of Problem (just a 1-2 line summary!):
Find the best Random Forest
"""

from prepare_data import RANDOM_STATE, VERBOSE, X_train_encoded, \
                         X_test_encoded, Y_train, Y_test
from sklearn.ensemble import RandomForestClassifier
from helper_functions import test_model
from sklearn.model_selection import GridSearchCV


def find_best_model():
    # Fit, predict, and compute accuracy/confusion matrix with the best
    # Random Forest model
    print(f"{' Random Forest ':=^100}")
    rf_model = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)
    # Select the hyperparameters to test
    hyper_params = {
        'n_estimators': [20, 50, 100, 100, 200, 500],
        'min_samples_leaf': [1, 5, 10],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [3, 5, None],
        'criterion': ["entropy"]
    }
    grid_search = GridSearchCV(estimator=rf_model,
                               param_grid=hyper_params,
                               n_jobs=-1, verbose=VERBOSE, scoring="accuracy")
    # Find the best RF model using the different hyperparameters
    grid_search.fit(X_train_encoded, Y_train)
    # Print the best parameters
    print(f"Best RF classifier hyperparameters: {grid_search.best_params_}")
    # Print the accuracy and confusion matrix
    test_model(grid_search.best_estimator_, X_train_encoded, X_test_encoded,
               Y_train, Y_test)
    print(f"{'=':=>100}", "\n")


if __name__ == "__main__":
    find_best_model()
