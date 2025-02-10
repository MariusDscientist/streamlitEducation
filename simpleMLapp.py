import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
 
This app predicts the iris flower type!  
""")

st.sidebar.header('User input parameters')


def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'Sepal_length': sepal_length,
            'Sepal_width': sepal_width,
            'Petal_length': petal_length,
            'Petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)
iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels')
classNames = pd.DataFrame(iris.target_names, columns=['Class Names'])
st.dataframe(classNames)

st.subheader('Prediction')
predictDF = pd.DataFrame(iris.target_names[prediction], columns=['Value'])
st.dataframe(predictDF)

st.subheader('Prediction Probability')
proba = pd.DataFrame(prediction_proba, columns=[
                     'Setosa', 'Versicolor', 'Virginica'])
st.dataframe(proba)
