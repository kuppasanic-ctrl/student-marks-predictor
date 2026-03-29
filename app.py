import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📊 Student Marks Predictor")

# Sample data (hours vs marks)
hours = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
marks = np.array([20, 40, 50, 70, 90])

# Train model
model = LinearRegression()
model.fit(hours, marks)

# User input
study_hours = st.number_input("Enter hours studied", min_value=0.0, max_value=24.0)

# Prediction
if st.button("Predict Marks"):
    prediction = model.predict([[study_hours]])
    st.success(f"Predicted Marks: {prediction[0]:.2f}")

# Plot graph
fig, ax = plt.subplots()
ax.scatter(hours, marks, color='blue')
ax.plot(hours, model.predict(hours), color='red')
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Marks")
ax.set_title("Student Marks Prediction")

st.pyplot(fig)