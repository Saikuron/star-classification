"""
Jean de Malliard
Class: CS 677
Date: 08/17/2022
Project
Description of Problem (just a 1-2 line summary!):
Find the best Decision tree
"""

from prepare_data import RANDOM_STATE, VERBOSE, X_train_encoded, \
                         X_test_encoded, Y_train, Y_test
from sklearn.tree import DecisionTreeClassifier, plot_tree
from helper_functions import test_model
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def find_best_model():
    # Fit, predict, and compute accuracy/confusion matrix with the best
    # decision tree model
    print(f"{' Decision Tree ':=^100}")
    tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    # Inspired by this link:
    # https://www.kaggle.com/code/gauravduttakiit/hyperparameter-tuning-in-decision-trees
    # Select the hyperparameters to test
    hyper_params = {
        'max_depth': [1, 2, 3, 4, 5, 10],
        'min_samples_leaf': [5, 10, 20, 30],
        'criterion': ["entropy"]
    }
    grid_search = GridSearchCV(estimator=tree_model,
                               param_grid=hyper_params,
                               n_jobs=-1, verbose=VERBOSE, scoring="accuracy")
    # Find the best DT using the different hyperparameters
    grid_search.fit(X_train_encoded, Y_train)
    # Print the best parameters
    print(f"Best DT classifier hyperparameters: {grid_search.best_params_}")
    # Print the accuracy and confusion matrix
    test_model(grid_search.best_estimator_, X_train_encoded, X_test_encoded,
               Y_train, Y_test)
    # If the file is ran by the user, print the tree as well
    if __name__ == "__main__":
        plot_tree(grid_search.best_estimator_)
        plt.show()
    print(f"{'=':=>100}", "\n")


if __name__ == "__main__":
    find_best_model()
