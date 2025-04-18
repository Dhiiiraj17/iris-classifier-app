import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Set up page config
st.set_page_config(
    page_title="🌸 Iris Classifier",
    page_icon="🌼",
    layout="centered"
)

# Load dataset and train model
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

# Centered layout
col1, col2, col3 = st.columns([0.2, 1, 0.2])

with col2:
    st.markdown("<h1 style='text-align: center;'>🌸 Iris Flower Classifier</h1>", unsafe_allow_html=True)
    st.image("https://i.imgur.com/SjE4VNO.png", width=120)

    st.markdown("#### 📲 Enter flower measurements:")

    with st.form("input_form"):
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")
        submit = st.form_submit_button("🌼 Predict")

    if submit:
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)[0]
        flower_name = iris.target_names[prediction]

        st.markdown("---")
        st.success(f"🎉 Predicted: **{flower_name.capitalize()}**")

        # Optional: Show emojis or mini "card" result
        st.markdown(f"<div style='text-align:center; font-size:24px;'>🌿 {flower_name.capitalize()} 🌿</div>", unsafe_allow_html=True)
