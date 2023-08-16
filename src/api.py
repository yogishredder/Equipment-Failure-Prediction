from fastapi import FastAPI, Request
import pandas as pd
import uvicorn
import util as utils
import data_preparation
import data_preprocessing
import joblib
from pydantic import BaseModel, Field

#Load config
config = utils.load_config()
model_data = utils.pickle_load('../' + config['best_model']['model_path'] + config['best_model']['model_name'])

class api_data(BaseModel):
    type : str = Field(alias='Type')
    air_temperature : float = Field(alias="Air temperature [K]")
    process_temperature : float = Field(alias="Process temperature [K]")
    rotational_speed : int = Field(alias="Rotational speed [rpm]")
    torque : float = Field(alias="Torque [Nm]")
    tool_wear : int = Field(alias="Tool wear [min]")

app = FastAPI()    

@app.get("/")
def home():
    return "Hello, FastAPI up!"

# predict data
@app.post("/predict/")
async def predict(data: api_data):  
      
    # load request
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)
  
    #Preprocess input data
    data.columns = config['dataset']["predictors"]
    
    #try: 
    data_dummy = data.copy()
    data_dummy = data_dummy.drop(config['dataset']['category_columns'], axis=1)
    data_label = data_preprocessing.encoding_category(data[config['dataset']['category_columns']])
    data_clean = pd.concat([data_label, data_dummy], axis=1)
    
    #predict data
       
    y_pred = model_data.predict(data_clean)

    if y_pred[0] == 0:
        y_pred = "Machine is normal."
    else:
        y_pred = "Machine is broken."
    return {"res" : y_pred, "error_msg": ""}


if __name__ == '__main__':
    uvicorn.run('api:app', host = '0.0.0.0', port = 8000, reload = True)