import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def load_data(train_file, test_file, label_column):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    

    y_train = train_df[label_column]
    y_test = test_df[label_column]
    

    X_train = train_df.drop(columns=[label_column])
    X_test = test_df.drop(columns=[label_column])
    
    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, X_test, y_test):
    encoders = {}
    valid_indices = X_test.index.tolist()  
    
    for col in X_train.columns:
        if X_train[col].dtype == 'object': 
            encoders[col] = LabelEncoder()
            X_train[col] = encoders[col].fit_transform(X_train[col])
            

            unseen_values = set(X_test[col]) - set(encoders[col].classes_)
            invalid_rows = X_test[X_test[col].isin(unseen_values)].index
            

            valid_indices = list(set(valid_indices) - set(invalid_rows))
            

            X_test = X_test.loc[valid_indices]  
            y_test = y_test.loc[valid_indices]
            X_test[col] = encoders[col].transform(X_test[col])

    return X_train, X_test, y_test


def train_and_evaluate(train_file, test_file, label_column):
    X_train, y_train, X_test, y_test = load_data(train_file, test_file, label_column)
    X_train, X_test, y_test = preprocess_data(X_train, X_test, y_test)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')


train_and_evaluate('output.csv', 'heloc.csv', 'RiskPerformance')