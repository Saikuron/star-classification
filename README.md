Running the code is easy

In essence, running the main.py file is enough. It will run the classifiers optimization and print the results for each classifier.

The prepare_data.py file extracts the data from the .csv file and preprocesses it. It is then used by the other files.
The helper_functions.py files contains some functions used by the classifiers files.
The files that have a name of classifier are using the data to train and find the best hyperparameters for the classifiers. Running any of those file will run the classifier optimization for this classifier.
