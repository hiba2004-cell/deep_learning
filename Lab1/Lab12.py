# Lab12 : Iris Classification.
# Realisé par : Jedata Rachid ENSAJ 2025-2026
# jedatarachid@gmail.com
# Ref: 

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import streamlit as st

# Step 1: Dataset
iris = datasets.load_iris()
print(iris.feature_names)
print(iris.data)
print(iris.target)
print(iris.target_names)

# Step 2: Model
models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1,random_state=42)
}

selected_model = st.sidebar.selectbox("Select your model", list(models.keys()))
model = models[selected_model]
# Step 3: Train
model.fit(iris.data, iris.target)
# Step 4: Test
# prediction = model.predict([[5, 2, 4, 3]])
# print(prediction)
# print(iris.target_names[prediction])

# Deploy your model
st.header('Classification of iris flowers !')
st.sidebar.header('Iris Features')
st.image("./images/iris1.png")


def user_input():
    sepal_length = st.sidebar.slider('sepal_length:', 4.3, 7.9, 6.0)
    sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider('petal_length:', 1.0, 9.2, 2.0)
    petal_width = st.sidebar.slider('petal_width', 0.1, 2.5, 1.0)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    flower_features = pd.DataFrame(data, index=[0])
    return flower_features


df = user_input()
st.write(' selected Model  :', selected_model)
st.subheader('iris flowers prediction:')
st.write(df)
prediction = model.predict(df)
st.subheader('iris flower category is:')
st.write(iris.target_names[prediction])
st.image("./images/iris_data.png")