import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="🌸 Advanced Iris Classifier", layout="wide")
st.title("🌸 Advanced Iris Flower Classifier")
st.write("Explore the Iris dataset, adjust inputs, and see model predictions with insights!")

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

st.sidebar.header("Input Flower Features")
sepal_length = st.sidebar.slider("Sepal length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()), float(X["sepal length (cm)"].mean()))
sepal_width = st.sidebar.slider("Sepal width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()), float(X["sepal width (cm)"].mean()))
petal_length = st.sidebar.slider("Petal length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()), float(X["petal length (cm)"].mean()))
petal_width = st.sidebar.slider("Petal width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()), float(X["petal width (cm)"].mean()))

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris.feature_names)
prediction = clf.predict(input_data)[0]
predicted_species = target_names[prediction]

st.subheader("🌼 Prediction")
st.success(f"The predicted species is: **{predicted_species.capitalize()}**")

st.subheader("📊 Feature Importance")
importance_df = pd.DataFrame({"Feature": iris.feature_names, "Importance": clf.feature_importances_})
fig_imp, ax_imp = plt.subplots()
sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax_imp)
st.pyplot(fig_imp)

st.subheader("🧾 Model Performance on Test Set")
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.background_gradient(cmap="Blues"))

cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

st.caption("Made with ❤️ using Streamlit")
