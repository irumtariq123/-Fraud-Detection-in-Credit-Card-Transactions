import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained Logistic Regression model
with open('logistic_regression_model.pkl', 'rb') as model_file:
    log_reg = pickle.load(model_file)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Load the feature columns
with open('feature_columns.pkl', 'rb') as feature_file:
    feature_columns = pickle.load(feature_file)

# Function to handle unseen labels
def handle_unseen_labels(label_encoder, data_column):
    known_classes = list(label_encoder.classes_)
    transformed_column = data_column.apply(lambda x: label_encoder.transform([x])[0] if x in known_classes else -1)
    return transformed_column

# Streamlit app layout
st.title('Fraud Detection Prediction')

# Input fields for new transaction data
transaction_id = st.text_input("Transaction ID", "")
customer_id = st.text_input("Customer ID", "")
merchant = st.text_input("Merchant", "")
location = st.text_input("Location", "")
transaction_date = st.date_input("Transaction Date", value=pd.to_datetime("2024-09-15"))
transaction_amount = st.number_input("Transaction Amount", min_value=0, value=500)
transaction_type = st.selectbox("Transaction Type", ['In-Store Purchase', 'Online Purchase','ATM Withdrawal'])
card_type = st.selectbox("Card Type", ['Visa', 'Discover','MasterCard', 'Amex'])

# New transaction data
new_transaction_data = {
    'Transaction ID': [transaction_id],
    'Customer ID': [customer_id],
    'Transaction Date': [transaction_date],
    'Transaction Amount': [transaction_amount],
    'Merchant': [merchant],
    'Location': [location],
    'Transaction Type': [transaction_type],
    'Card Type': [card_type],
}

# Convert the new transaction features to a DataFrame
new_transaction_df = pd.DataFrame(new_transaction_data)

# Label Encoding for 'Transaction ID' and 'Customer ID'
for col in ['Transaction ID', 'Customer ID']:
    new_transaction_df[col] = handle_unseen_labels(label_encoders[col], new_transaction_df[col])

# Convert 'Transaction Date' to numerical format (extract features)
new_transaction_df['Transaction Date'] = pd.to_datetime(new_transaction_df['Transaction Date'])
new_transaction_df['Transaction Year'] = new_transaction_df['Transaction Date'].dt.year
new_transaction_df['Transaction Month'] = new_transaction_df['Transaction Date'].dt.month
new_transaction_df['Transaction Day'] = new_transaction_df['Transaction Date'].dt.day
new_transaction_df = new_transaction_df.drop('Transaction Date', axis=1)

# Apply one-hot encoding to the categorical columns
new_transaction_encoded = pd.get_dummies(new_transaction_df, columns=['Merchant', 'Location', 'Transaction Type', 'Card Type'], drop_first=True)

# Align the new transaction DataFrame with the training set columns (important to handle missing columns)
new_transaction_encoded = new_transaction_encoded.reindex(columns=feature_columns, fill_value=0)

# Button to trigger prediction
if st.button("Predict Fraud"):
    # Predict using the trained Logistic Regression model
    log_reg_prediction = log_reg.predict(new_transaction_encoded)
    # Predict the fraud probability for the new transaction
    log_reg_probability = log_reg.predict_proba(new_transaction_encoded)

    # Set a custom threshold for fraud detection (e.g., 0.7 instead of the default 0.5)
    fraud_threshold = 0.7

    # Make the prediction based on the threshold
    fraud_prediction = 1 if log_reg_probability[0][1] >= fraud_threshold else 0

    # Display results
    st.write("**Logistic Regression Fraud Prediction:**", "Fraud" if fraud_prediction == 1 else "Not Fraud")
    st.write("**Logistic Regression Fraud Probability:**", log_reg_probability[0][1])
