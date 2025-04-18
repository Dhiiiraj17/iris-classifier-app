import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Set page config for mobile-friendly experience
st.set_page_config(page_title="Iris Classifier 🌸", page_icon="🌼", layout="centered")

# Load data and train model
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

# Centered layout for mobile
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title("🌸 Iris Flower Classifier")
    st.image("https://i.imgur.com/SjE4VNO.png", width=150)  # Replace with your own image link if you want

    st.markdown("### 📲 Enter flower details:")

    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

    if st.button("Predict"):
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)[0]
        flower_name = iris.target_names[prediction]
        st.success(f"🌼 Predicted: **{flower_name.capitalize()}**")
