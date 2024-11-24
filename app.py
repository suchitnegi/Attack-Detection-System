import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import subprocess

# Load trained models and preprocessing tools
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
knn_model = joblib.load('knn_model.pkl')
svm_model = joblib.load('svm_model.pkl')

scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define function for predictions
def predict_attack(model, data):
    predictions = model.predict(data)
    predicted_labels = label_encoder.inverse_transform(predictions)
    return predicted_labels

# Function to block IP in Windows Firewall
def block_ip(ip_address):
    try:
        # Command to add a firewall rule to block the IP
        command = f'netsh advfirewall firewall add rule name="Block {ip_address}" dir=in action=block protocol=TCP remoteip={ip_address} enable=yes'
        subprocess.run(command, check=True, shell=True)
        st.write(f"Successfully blocked IP: {ip_address}")
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to block IP: {ip_address}. Error: {str(e)}")

# Streamlit App UI
st.title("Network Attack Detection")

st.write("### Download Sample File")
# Provide download button for the sample file
sample_file = 'combinedsampledata.csv'

with open(sample_file, "rb") as f:
    st.download_button(
        label="Download Sample CSV File",
        data=f,
        file_name=sample_file,
        mime="text/csv"
    )

st.write("### Upload Your CSV File")
# Allow users to upload their file
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

st.write("### Training Performance Evaluation of Various Models")
comparisondf = pd.read_csv('comparisonintrainwithsvm.csv')
st.dataframe(comparisondf, width=1000)

if uploaded_file:
    # Read uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(data)

    # Define required columns
    selected_columns = [
        'Bwd Packet Length Max', 'Fwd Packet Length Max', 'Average Packet Size',
        'Max Packet Length', 'Total Backward Packets', 'Total Fwd Packets',
        'Destination Port', 'Min Packet Length', 'Flow Bytes/s',
        'Bwd Header Length', 'Fwd Header Length'
    ]

    # Check for missing columns
    missing_columns = [col for col in selected_columns if col not in data.columns]
    if missing_columns:
        st.error(f"The following columns are missing: {', '.join(missing_columns)}")
    else:
        # Extract and preprocess features
        X = data[selected_columns]
        X_scaled = scaler.transform(X)

        # Dropdown to select the model
        model_choice = st.selectbox(
            "Choose a Model",
            ("Random Forest", "XGBoost", "KNN", "SVM")
        )

        # Select model
        if model_choice == "Random Forest":
            selected_model = rf_model
        elif model_choice == "XGBoost":
            selected_model = xgb_model
        elif model_choice == "SVM":
            selected_model = svm_model
        else:
            selected_model = knn_model

        # Predict attacks
        if st.button("Detect Attacks"):
            predictions = predict_attack(selected_model, X_scaled)

            # Add predictions to the original data
            data['Attack Type'] = predictions

            # Filter rows with attacks
            attacks = data[data['Attack Type'] != 'BENIGN']
            st.write("### Attacks Detected")
            if not attacks.empty:
                st.dataframe(attacks[['IP Address', 'Attack Type']], width=1000)

                # Block attacker IPs
                for ip in attacks['IP Address'].unique():
                    block_ip(ip)  # Block each detected attacker's IP

            else:
                st.write("No attacks detected.")

            # Calculate accuracy of the selected model
            label_file = 'combinedsampledatalabel.csv'
            label_data = pd.read_csv(label_file)
            if len(label_data) != len(predictions):
                st.error("Mismatch between uploaded data and label data.")
            else:
                y_true = label_data['Label']
                y_pred = label_encoder.transform(predictions)

                # Calculate accuracy
                accuracy = accuracy_score(y_true, y_pred)
                st.write(f"### Accuracy of {model_choice}: {accuracy * 100:.2f} %")
