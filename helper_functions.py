"""
Jean de Malliard
Class: CS 677
Date: 08/17/2022
Project
Description of Problem (just a 1-2 line summary!):
Helper functions
"""

from sklearn.metrics import confusion_matrix, accuracy_score

# Number of classes used by the dataset (used for the confusion matrix)
NB_CLASSES = 6


# I made this function but didn't use it in the end
def compute_accuracies(test_set, predictions, nb_classes):
    # Modified version of the compute_accuracies function, this time there
    # can be as many classes as needed
    cnt_right = 0
    cnt_wrong = 0
    conf_matrix = [[0 for x in range(nb_classes)] for y in range(nb_classes)]
    # For every row of the test/prediction set, increment the appropriate
    # value of the confusion matrix
    for i in range(len(test_set)):
        conf_matrix[test_set[i]][predictions[i]] += 1
        # Count the correctly/incorrectly classified tuples to compute accuracy
        if predictions[i] == test_set[i]:
            cnt_right += 1
        else:
            cnt_wrong += 1
    # Once each row has been counted, we return accuracy and confusion matrix
    accuracy = round(cnt_right / (cnt_right + cnt_wrong), 4)
    return (accuracy, conf_matrix)


def print_confusion_matrix(conf_matrix):
    # Modified version of the print_confusion_matrix but it can print the
    # confusion matrix for as many classes as needed
    classes = [x for x in range(len(conf_matrix))]
    row_format = "{:>12}" * (len(classes) + 1)
    print(f"{'Predicted class':>36}")
    print(row_format.format("Actual class", *classes))
    for label, row in zip(classes, conf_matrix):
        print(row_format.format(label, *row))


def test_model(model, X_train_, X_test_, Y_train_, Y_test_):
    # Function to predict values, print the accuracy and confusion matrix
    # Used to evaluate a trained model
    preds_train = model.predict(X_train_)
    preds_test = model.predict(X_test_)
    # Compute train and test accuracy
    model_accuracy_train = accuracy_score(Y_train_, preds_train)
    model_accuracy_test = accuracy_score(Y_test_, preds_test)
    # Compute confusion matrix
    model_conf = confusion_matrix(Y_test_, preds_test)
    # Print accuracies and confusion matrix
    print(f"Training Accuracy: {model_accuracy_train}")
    print(f"Testing  Accuracy: {model_accuracy_test}")
    print(f"Confusion matrix:\n")
    print_confusion_matrix(model_conf)
