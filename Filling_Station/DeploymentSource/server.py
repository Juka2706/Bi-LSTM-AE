import sys, os, inspect
sys.path.append('../../')
import json
from fastapi import FastAPI
import numpy as np
import socket

#Set parent directory to sys path to find the library in parent folder directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import timeserieslibrary.ModelDeployment as MD
import timeserieslibrary.anomali_detection as MA
import timeserieslibrary.Preprocessing as pp

app = FastAPI()



with open("..\config.json", "r") as file:
    data = json.load(file)

    number_of_variables=len(list(data['index_list_of_used_variablen'].values()))
    threshold = list(data['thresholds'].values())        
    downsampling_faktor= data['downsampling_faktor']
    input_window=data['split_time_series_into_window_of_len']
    sliding_window_stride=data['sliding_window_stride_deployment']
    file.close()


ts_processing = pp.TimeSeriesPreprocessingPipeline(JSON_config_file="..\config.json")


model_path = '../Model'
model_name = ''


model = MD.TFModel(model_path,  batch_size=1)
anomalie_detector = MA.MultiVariantMSEAnomalieDetection(threshold = threshold)
anomalie_detection = MA.AnomalieDetection(ts_processing, model, anomalie_detector, (input_window*downsampling_faktor), number_of_variables,  sliding_window_stride)


#post. put. get. delete
@app.get("/")
async def hello():
    return {"message": "hello world"}

@app.get("/getdata") 
async def getdata():# -> list[list[list[float]]]:
    
    data =  anomalie_detection.read(message_type = "GetData" )
    return data

    
#input_data:    Data for one time step shape: (number_of_variables) 
@app.post("/pushdata")
async def pushdata(input_data:list[list[float]]):   #list[list[float]]  or list[float] mandotory

    print(np.array(input_data).shape)
    result = anomalie_detection.write( data = input_data, message_type = "InputData" )

    return result









