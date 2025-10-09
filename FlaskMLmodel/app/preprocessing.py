import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess dataset
def preprocess_data(df):
    # Handle missing values
    df.ffill(inplace=True)

    # Discretization (simplifying marital status)
    df['marital-status'] = df['marital-status'].replace(
        ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
         'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
        ['divorced', 'married', 'married', 'married',
         'not married', 'not married', 'not married']
    )

    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Dropping redundant columns if they exist
    for col in ['fnlwgt', 'educational-num']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df, label_encoders

def split_data(df, target_column):
    X = df.drop(target_column, axis=1).values
    Y = df[target_column].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    return X_train, X_test, Y_train, Y_test

# Train model
def train_model(X_train, Y_train):
    clf_gini = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth= 5, min_samples_leaf=5)
    clf_gini.fit(X_train, Y_train)
    return clf_gini

# Evaluate model
def evaluate_model(clf_gini, X_test, Y_test):
    Y_pred = clf_gini.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred)
    matrix = confusion_matrix(Y_test, Y_pred)
    return accuracy, report, matrix

# Example workflow
if __name__ == "__main__":
    file_path = "FlaskMLmodel/app/adult.csv"  # fixed variable name
    df = load_data(file_path)
    df, label_encoders = preprocess_data(df)
    target_column = 'income'  # Change if your target column has a different name
    X_train, X_test, Y_train, Y_test = split_data(df, target_column)
    clf_gini = train_model(X_train, Y_train)
    accuracy, report, matrix = evaluate_model(clf_gini, X_test, Y_test)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    # Save trained model
    with open("model.pkl", "wb") as model_file:
        pickle.dump(clf_gini, model_file)



## FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
 # df.fillna(method='ffill', inplace=True)
