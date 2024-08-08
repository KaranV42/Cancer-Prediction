import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle as pickle

def get_the_data():
    df = pd.read_csv('data/cancer_data.csv')
    df.drop(columns=['Unnamed: 32'], inplace=True)
    df.replace(to_replace={'diagnosis': {'M': 0, 'B': 1}}, inplace=True)
    
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    return X, y


def create_a_model(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X=X)

    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, stratify=y, random_state=69)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, scaler


def main():

    X, y = get_the_data()
    model, scaler = create_a_model(X, y)

    with open('model/model.pkl','wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':

    main()