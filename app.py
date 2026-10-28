
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("iris_model.pkl")

# Species names
species_names = ["setosa", "versicolor", "virginica"]

st.title("Iris Flower Classification 🌸")
st.write("Predict the species of an Iris flower using its measurements.")

# User inputs
sepal_length = st.number_input(
    "Sepal length (cm)", min_value=0.0, value=5.1
)
sepal_width = st.number_input(
    "Sepal width (cm)", min_value=0.0, value=3.5
)
petal_length = st.number_input(
    "Petal length (cm)", min_value=0.0, value=1.4
)
petal_width = st.number_input(
    "Petal width (cm)", min_value=0.0, value=0.2
)

# Predict species
if st.button("Predict Species"):
    features = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    prediction = model.predict(features)[0]
    st.success(f"Predicted species: {species_names[prediction].title()}")
