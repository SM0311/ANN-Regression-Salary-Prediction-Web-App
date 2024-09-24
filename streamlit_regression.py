import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np


# Load the trained model and preprocessors
model = tf.keras.models.load_model('regression_model.h5')
with open('lable_regression_encoder_gender.pkl', 'rb') as file:
    label_regression_encoder_gender = pickle.load(file)

with open('onehot_regression_encoder_geo.pkl', 'rb') as file:
    onehot_regression_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set page configuration
st.set_page_config(page_title="Customer Salary Prediction", page_icon="💰", layout='wide')

# Custom CSS for styling
st.markdown("""
<style>
.main { background-color: #F0F2F5; }
h1 { color: #F63366; text-align: center; }
.header { font-size: 2.8em; margin-bottom: 0.5em; font-weight: bold; }
.subheader { text-align: center; color: #6c757d; font-size: 1.3em; margin-bottom: 1em; }
.section-header { color: #007BFF; font-size: 1.6em; margin-top: 0.5em; border-bottom: 2px solid #007BFF; padding-bottom: 0.5em; }
.prediction { color: #28A745; font-size: 2.2em; font-weight: bold; text-align: center; margin-top: 1.5em; }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown("<h1 class='header'>Customer Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Input customer details to predict their salary.</p>", unsafe_allow_html=True)

# Create a form for user input
with st.form(key='salary_prediction_form'):
    st.markdown("<h2 class='section-header'>Customer Information</h2>", unsafe_allow_html=True)
    geography = st.selectbox('Geography', onehot_regression_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_regression_encoder_gender.classes_)
    age = st.number_input('Age', min_value=18, max_value=92, value=30)

    st.markdown("<h2 class='section-header'>Banking Details</h2>", unsafe_allow_html=True)
    tenure = st.slider('Tenure (Years)', 0, 10, value=5)
    balance = st.number_input('Account Balance', min_value=0.0, value=1000.0, step=100.0)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
    num_of_products = st.slider('Number of Products', 1, 4, value=2)
    has_cr_card = st.selectbox('Credit Card Holder', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    is_active_member = st.selectbox('Active Member', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    exited = st.selectbox('Exited the Bank', [0, 1], format_func=lambda x: 'Yes' if x else 'No')

    submit_button = st.form_submit_button(label='Predict Salary')

# When the form is submitted, make the prediction
if submit_button:
    st.markdown("### Predicting salary...")
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_regression_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited]
    })
    
    geo_encoded = onehot_regression_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_regression_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_data_scaled)
    predicted_salary = prediction[0][0]
    st.markdown("<h2 class='prediction'>Estimated Salary: $ {}</h2>".format(round(predicted_salary, 2)), unsafe_allow_html=True)
    st.success('Prediction successful! 🎉')
    st.balloons()

# Sidebar for additional information and input ranges
st.sidebar.header("About This App")
st.sidebar.markdown("""
This app predicts a customer’s estimated salary based on personal and banking details.
- **Model:** Built using TensorFlow and Keras.
""")

st.sidebar.header("Input Ranges")
st.sidebar.markdown("""
- **Age:** 18 - 92
- **Tenure:** 0 - 10 years
- **Account Balance:** Min: $0.00
- **Credit Score:** 300 - 850
- **Products:** 1 - 4
- **Credit Card Holder:** Yes (1) / No (0)
- **Active Membership:** Yes (1) / No (0)
- **Exited:** Yes (1) / No (0)
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by [Suraj Mishra]")
