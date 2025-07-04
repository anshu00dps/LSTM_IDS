# LSTM-based Intrusion Detection System (IDS)

## Overview
This project implements an Intrusion Detection System (IDS) using Long Short-Term Memory (LSTM) networks to detect network intrusions and malicious activities. The system is trained on network traffic data to classify between normal and malicious network behaviors.

## Features
- Data preprocessing and feature engineering for network traffic data
- LSTM-based deep learning model for intrusion detection
- Model evaluation metrics including accuracy, precision, recall, and F1-score
- Confusion matrix visualization for performance analysis
- Trained on real-world network traffic data

## Requirements
- Python 3.6+
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LSTM-IDS.git
   cd LSTM-IDS
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: If requirements.txt doesn't exist, install the packages listed in the Requirements section.

## Dataset
This project uses the following dataset:
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

The dataset contains network traffic features for intrusion detection, including both normal and malicious traffic patterns.

## Usage
1. Open the Jupyter notebook `LSTM_IDS.ipynb`
2. Run the cells sequentially to:
   - Load and preprocess the data
   - Train the LSTM model
   - Evaluate the model performance
   - Visualize the results

## Model Architecture
The LSTM model consists of:
- LSTM layer with 64 units
- Dropout layer for regularization
- Dense output layer with sigmoid activation for binary classification

## Results
The model's performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC score

Confusion matrix visualization provides insights into the model's classification performance.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset provided by the Canadian Institute for Cybersecurity
- Built using TensorFlow and Keras
- Inspired by research in network security and deep learning
