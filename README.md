# CPI-demo

This is a demo created for CPI. It includes the code used to build a streamlit app (https://docs.streamlit.io/).
Files are as follows: 
- digi_twin_interface.py - contains the code used to build the app
- requirements.txt - these are the python packages required to run the app
- sample_ECG_data_and_classification.zip - zip file containing example patient ECG data in the format patientID.csv (patient data taken from the publicly available MIT-BIH arrythmia database: https://www.physionet.org/content/mitdb/1.0.0/), and model predictions from a CNN classifier in the format patientID.npy.

The csv and npy files are used by digi_twin_interface.py to visualise patient ECGs and abnormal segments as identified by the model. 

This app can be deployed on your own servers using streamlit, or using GitHub codespaces. 

Questions can be emailed to kanberelise[at]gmail[dot]com 


