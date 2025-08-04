# 311_predictions

This project is the model creation of a prediction model from the NYC 311 data to predict whether a complaint will be resolved within 7 days. For this the chosen alorithm is XGBoost. 

It highlights the usage of MLFlow for model storage as well as EvidentlyAI for monitorinfg. 

This is the capstone project for the Datatalks MLOps course. 


MAIN URL with explanations, schema, etc. is found at: 
https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/about_data

The API URL is: https://data.cityofnewyork.us/resource/erm2-nwe9.csv
You do not need an API key for this public dataset.

Execution of the train_test_monitor.py script will produce an EvidentlyAI report in HTML format, as well as store the model in MLFlow.
The evidently report can be found in the same location as the python script; MLFlow can be accessed via the mlflow ui command.
