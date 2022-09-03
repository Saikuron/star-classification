"""
Jean de Malliard
Class: CS 677
Date: 08/17/2022
Project
Description of Problem (just a 1-2 line summary!):
Find the best SVM model
"""

from prepare_data import VERBOSE, X_train_encoded_scaled, \
                         X_test_encoded_scaled, Y_train, Y_test
from sklearn.svm import SVC
from helper_functions import test_model
from sklearn.model_selection import GridSearchCV


def find_best_model():
    # Fit, predict, and compute accuracy/confusion matrix with the best
    # SVM model
    print(f"{' SVM ':=^100}")
    svm_model = SVC()
    # Select the hyperparameters to test
    hyper_params = {
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(estimator=svm_model,
                               param_grid=hyper_params,
                               n_jobs=-1, verbose=VERBOSE, scoring="accuracy")
    # Find the best SVM model using the different hyperparameters
    grid_search.fit(X_train_encoded_scaled, Y_train)
    # Print the best parameters
    print(f"Best SVM classifier hyperparameters: {grid_search.best_params_}")
    # Print the accuracy and confusion matrix
    test_model(grid_search.best_estimator_, X_train_encoded_scaled,
               X_test_encoded_scaled, Y_train, Y_test)
    print(f"{'=':=>100}", "\n")


if __name__ == "__main__":
    find_best_model()
