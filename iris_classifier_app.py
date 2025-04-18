import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt


# Load Iris data
iris = load_iris()
# Convert to DataFrame for plotting
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

X = iris.data
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🌸 Iris Flower Predictor")
st.write("Enter the flower measurements below to predict the species.")

# User input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_species = iris.target_names[prediction[0]]

st.subheader("🌼 Prediction:")
st.success(f"The predicted iris species is **{predicted_species}**.")
st.subheader("📊 Petal Length vs Petal Width Scatter Plot")

fig, ax = plt.subplots()

# Plot all points by species
for species in iris.target_names:
    subset = df[df['species'] == species]
    ax.scatter(
        subset['petal length (cm)'],
        subset['petal width (cm)'],
        label=species
    )

# Plot the user input point in red
ax.scatter(petal_length, petal_width, color='red', s=100, label="Your Flower", marker="X")

ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_title("Iris Dataset - Petal Length vs Width")
ax.legend()

st.pyplot(fig)

