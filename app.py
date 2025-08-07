

import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.set_page_config(page_title="Iris Flower Classifier", layout="centered")

st.title("🌸 Iris Flower Classifier")
st.write("Choose flower properties to predict the species")

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
clf = RandomForestClassifier()
clf.fit(X, y)

# Sidebar sliders
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()), float(X["sepal length (cm)"].mean()))
sepal_width = st.sidebar.slider("Sepal width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()), float(X["sepal width (cm)"].mean()))
petal_length = st.sidebar.slider("Petal length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()), float(X["petal length (cm)"].mean()))
petal_width = st.sidebar.slider("Petal width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()), float(X["petal width (cm)"].mean()))

# Prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris.feature_names)
prediction = clf.predict(input_data)[0]
predicted_species = iris.target_names[prediction]

st.subheader("Prediction Result")
st.success(f"The predicted species is: **{predicted_species.capitalize()}** 🌼")
