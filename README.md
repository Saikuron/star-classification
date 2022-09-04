# :star: Star classification :star:

## About

This project was made for my *Data Science with Python* class while I was studying for my Master of Science in Computer Information Systems at Boston University. \
 \
The goal was to build different models using different data science algorithms and select the best model, in order to classify stars as being either:
- Red Dwarves
- Brown Dwarves
- White Dwarves
- Main Sequence
- Super Giants
- Hyper Giants

## Usage

The projects consists of the dataset file, a few python files, a report file, and this README file.

The python files having a type of classifier as the filename are using the data to train and find the best hyperparameters for a specific type of classifier. Running any of those file will run the classifier optimization for this classifier. For example:
```
python knn.py
```

Running the main.py file will run the classifiers optimization files and print the results for each classifier:
```
python main.py
```

The prepare_data.py file extracts the data from the .csv file and preprocesses it. It is then used by the other files. \
The helper_functions.py files contains some functions used by the classifiers files. \
