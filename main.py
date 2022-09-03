"""
Jean de Malliard
Class: CS 677
Date: 08/18/2022
Project
Description of Problem (just a 1-2 line summary!):
Run all the classifiers
"""

import svm
import knn
import naive_bayes
import decision_tree
import random_forest

# Run the different classifiers and find the best parameters for each one
if __name__ == "__main__":
    svm.find_best_model()
    knn.find_best_model()
    naive_bayes.find_best_model()
    decision_tree.find_best_model()
    random_forest.find_best_model()
