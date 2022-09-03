"""
Jean de Malliard
Class: CS 677
Date: 08/17/2022
Project
Description of Problem (just a 1-2 line summary!):
Open and read the data from the original dataset
Fix, split, encode and scale the data, so other files can use whatever they
need
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants used throught the files
SPLIT_SIZE = 0.5
RANDOM_STATE = 10
VERBOSE = 0

# Construct the filename and its path
filename = "dataset_original"
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
filename_full = os.path.join(input_dir, filename + '.csv')
# Load the file data into a dataframe
try:
    df = pd.read_csv(filename_full)
    print('File fed to a dataframe: ', filename)

except Exception as e:
    print(e)
    print('failed to read data for: ', filename)


# The Color attribute has non-consistent values
def fix_color(color):
    if color == "white-yellow":
        return "yellow-white"
    if color == "blue white":
        return "blue-white"
    if color == "pale yellow orange":
        return "pale-yellow-orange"
    if color == "yellowish white":
        return "yellowish-white"
    else:
        return color


# Fix the color problems
df["Color"] = df["Color"].str.lower()
df["Color"] = df["Color"].apply(fix_color)

# Split features and class
X = df.drop("Type", axis=1)
Y = df.loc[:, ('Type')]
# Make a training and testing dataset using the base dataset
# No encoding, no scaling
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=SPLIT_SIZE, random_state=RANDOM_STATE)

# Encode nominal attributes in binary variables
nominal_data = df[["Color", "Spectral_Class"]]
dummies = [pd.get_dummies(df[c]) for c in nominal_data]
dummies = pd.concat(dummies, axis=1)
df_encoded = pd.concat([df, dummies], axis=1).drop(
    ["Color", "Spectral_Class"], axis=1)
# Separate features and class for the encoded dataset
X_encoded = df_encoded.drop("Type", axis=1)
# Make a training and testing dataset using the encoded dataset
# Encoding, no scaling
X_train_encoded, X_test_encoded, Y_train, Y_test = train_test_split(
    X_encoded, Y, train_size=SPLIT_SIZE, random_state=RANDOM_STATE)

# Scale the data
# Scaling, no encoding
X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(
    X, Y, train_size=SPLIT_SIZE, random_state=RANDOM_STATE)
# Scale the numeric attributes
columns_to_scale = ['Temperature', 'L', 'R', 'A_M']
scaler = StandardScaler()
scaler.fit(X_train_scaled[columns_to_scale])
X_train_scaled[columns_to_scale] = scaler.transform(
    X_train_scaled[columns_to_scale])
X_test_scaled[columns_to_scale] = scaler.transform(
    X_test_scaled[columns_to_scale])
# Rebuild the datasets with scaled data, can be useful
X_scaled = pd.concat([X_train_scaled, X_test_scaled]).sort_index()
df_scaled = pd.concat([X_scaled, Y], axis=1)
# Make a dataset encoded and scaled
X_encoded_scaled = X_encoded.copy()
X_encoded_scaled[columns_to_scale] = X_scaled[columns_to_scale]
# Make a training and testing dataset using the encoded and scaled dataset
X_train_encoded_scaled, X_test_encoded_scaled, \
    Y_train, Y_test = train_test_split(
        X_encoded_scaled, Y, train_size=SPLIT_SIZE, random_state=RANDOM_STATE)

df_encoded_scaled = pd.concat([X_encoded_scaled, Y], axis=1)
