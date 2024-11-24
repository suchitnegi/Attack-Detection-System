# Network Attack Detection System

This is a **Network Attack Detection System** that uses machine learning models to detect various types of network attacks from user-uploaded CSV data. The system utilizes different models such as **Random Forest**, **XGBoost**, **KNN**, and **SVM** to predict attack types.

The app also provides functionality to download a sample file and upload custom data for attack detection. Additionally, the app blocks IP addresses of any detected attackers using the Windows Firewall.

---

## Features

- **Download Sample CSV File**: Users can download a sample file (`combinedsampledata.csv`) containing sample network data for testing.
- **Upload Custom CSV Data**: Users can upload their own CSV file to perform network attack detection.
- **Model Selection**: Choose from 4 different models to detect attacks: **Random Forest**, **XGBoost**, **KNN**, and **SVM**.
- **IP Blocking**: The app blocks the IP addresses of detected attackers using the Windows Firewall.
- **Accuracy Calculation**: Displays the accuracy of the selected model based on labeled data.

---

## Technologies Used

- **Streamlit**: For building the interactive web app.
- **Scikit-learn**: For machine learning models like Random Forest, KNN, and SVM.
- **XGBoost**: For the XGBoost model.
- **Joblib**: For loading pre-trained models and scaler.
- **Pandas & Numpy**: For data handling and manipulation.
- **Subprocess**: To interact with the Windows Firewall to block attacker IPs.

---

## Installation

### Prerequisites

1. **Python 3.x**: Ensure you have Python 3 installed on your local machine.
2. **Required Python Libraries**: The required libraries for this app are listed in the `requirements.txt` file.

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Attack-Detection-System.git
   cd Attack-Detection-System
