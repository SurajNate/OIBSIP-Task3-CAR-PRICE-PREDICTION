import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")
# Set the title of the Streamlit app
st.title("Car Price Prediction with Machine Learning - [@suraj_nate](https://www.instagram.com/suraj_nate/) 👀")

st.write("""
    This application predicts car prices based on various input features such as the car's age, mileage, fuel type, and more. 
    The model has been trained on a comprehensive dataset, ensuring accurate and reliable predictions.
    The dataset used for this project is sourced from Kaggle, providing detailed information on car specifications and their respective market prices.

    You can explore the dataset further on Kaggle :
    - Car Price Prediction Dataset [https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars]
""")
st.write("<hr>", unsafe_allow_html=True)

# Load the dataset locally
file_path = "Task 3 - Jupyter Files\\car data.csv" 
car_data = pd.read_csv(file_path)

# Load the pre-trained model
model = joblib.load("Task 3 - Jupyter Files\\car_price_prediction_surajnate_model.pkl")

# Predict Car Price for Unseen Data
st.header("Predict the Price of a Car")
st.write("Provide the necessary details to predict the car's selling price :")

# Inputs for prediction
fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG'], help="Select the fuel type of the car.")
selling_type = st.selectbox("Selling Type", options=['Dealer', 'Individual'], help="Choose whether the car is being sold by a dealer or an individual.")
transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'], help="Select whether the car has a Manual or Automatic transmission.")
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1, help="Enter the current price of the car in lakhs.")
driven_kms = st.number_input("Kilometers Driven", min_value=0, step=100, help="Enter the total number of kilometers the car has been driven.")
owner = st.selectbox("Number of Previous Owners", options=[0, 1, 2, 3], help="Specify how many previous owners the car has had.")
car_age = st.number_input("Car Age (in years)", min_value=0, step=1, help="Enter the age of the car in years.")

# Prepare unseen data for prediction (no manual encoding needed)
unseen_data = pd.DataFrame({
    'Fuel_Type': [fuel_type],
    'Selling_type': [selling_type],
    'Transmission': [transmission],
    'Present_Price': [present_price],
    'Driven_kms': [driven_kms],
    'Owner': [owner],
    'Car_Age': [car_age]
})

# Checking the data before prediction
st.write("Unseen Data:", unseen_data)

# Predict button
if st.button("Predict Price"):
    try:
        # prediction using the model
        prediction = model.predict(unseen_data)
        st.success(f"Predicted Selling Price: **₹{prediction[0]:.2f}** lakhs")
    except Exception as e:
        st.error(f"Error occurred: {e}")

# Data Loading and Exploration
st.header("Load and Explore the Dataset")

st.subheader("Dataset Preview")
st.dataframe(car_data.head())

st.subheader("Dataset Description")
st.write(car_data.describe())

# Derive 'Car_Age' from 'Year' and drop unnecessary columns
car_data['Car_Age'] = 2025 - car_data['Year']
car_data = car_data.drop(columns=['Year', 'Car_Name'])

# Display updated dataset
st.subheader("Updated Dataset (after preprocessing)")
st.dataframe(car_data.head())

# Footer
st.write("---")
st.markdown('<center><a href="https://www.instagram.com/suraj_nate/" target="_blank" style="color:white;text-decoration:none">&copy; 2025 @suraj_nate All rights reserved.</a></center>', unsafe_allow_html=True)