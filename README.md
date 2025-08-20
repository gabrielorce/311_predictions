# 311_predictions

This is the description of the capstone project for the DatataDataTalks.Club MLOps-Zoomcamp course.  


This project is the model creation of a prediction model from the NYC 311 data to predict the probability tahat a complaint will be resolved within 7 days. For this the chosen alorithm is XGBoost.  

It highlights the usage of MLFlow for model storage as well as EvidentlyAI for monitoring.  


The data is openly available in the NYC Open Data page. It can also be retrieved via API.

The main URL for the data source, including explanations, schema, API information and other items can be found at:   
https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/about_data

You can download a bulk csv file containing all the data from 2010 to present, but that is a very large file.  
So I only downloaded the data for 2023, which was used to train the model. 

The API URL is: https://data.cityofnewyork.us/resource/erm2-nwe9.csv  
(You do not need an API key for this public dataset)
I used this API to download only the current date information. The model makes the predictions based on this input data.


The csv files contains many values, but the ones selected as features used for the model's training and prediction are:  
- complaint_type  
- borough  
- agency  
- incident_zip  
  

Execution of the train_test_monitor.py script will produce an EvidentlyAI report in HTML format, as well as store the model in MLFlow.
The evidently report can be found in the same location as the python script.



PREPARATION:  
Be sure to install needed packages:

``` > pip install -r requirements.txt ```

A docker compose file includes MLFLow, Prometheus, Grafana and the predictor model's API.

to create the containers, the command is:  

``` > docker compose up ```

(in the same directory as the docker-compose.yml)  

USAGE:  
if using locally, the components can be accessed as follow:  
Via Web:  
[http://localhost:3000/](http://localhost:3000/)  - GRAFANA   
[http://localhost:5000/](http://localhost:5000/)  - MLFLOW   

[http://localhost:9000/](http://localhost:9000/)  - PROMETHEUS   

via API:   
[http://localhost:5000/](http://localhost:8080/predict)  -  PREDICTION MODEL 


Also, file export_metrics.py exposes these metrics and loads them into prometheus:  
```
PSI (psi_complaint_type)
ROC AUC (roc_auc_metric)
```

Be sure to install the prometheus_client before using it:   
```   
pip install prometheus_client  
python export_metrics.py  
```   

which you can view in:  
http://localhost:9100/metrics  


To make a prediction you can use curl once you have the feature information from table provided on the website. Below is an example:

```
 curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"complaint_type": "Noise - Street/Sidewalk", "borough": "BRONX", "agency": "NYPD", "incident_zip": 10452} 
```




NOTE: Unfortunately, this project currently lacks proper structure due to lack of time in completion. 
