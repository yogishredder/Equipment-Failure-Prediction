import pandas as pd
import util as utils
import copy
from sklearn.model_selection import train_test_split

#Load raw data
def read_raw_data(config: dict) -> pd.DataFrame:
    # Return raw dataset
    return pd.read_csv('../' + config['dataset']['data_directory']+ '/' + config['dataset']['file_name'])

#Drop column 
def drop_column(input_data: pd.DataFrame, config: dict):
    input_data = input_data.drop(config['dataset']['drop_columns'], axis=1)

    return input_data

#Handling missing data
def drop_rows_with_missing_data(input_data: pd.DataFrame, config: dict):
    input_data.dropna()
    
    return input_data

#Data Defense
def check_data(input_data: pd.DataFrame, config: dict, api: bool = False):
    input_data = copy.deepcopy(input_data)
    config = copy.deepcopy(config)

    len_input_data = len(input_data)

    # Check data types
    assert input_data.select_dtypes("int").columns.to_list() == \
        config['dataset']["int_columns"], "an error occurs in int column(s)."
    assert input_data.select_dtypes("float64").columns.to_list() == \
        config['dataset']["float_columns"], "an error occurs in float column(s)."
    assert input_data.select_dtypes("object").columns.to_list() == \
        config['dataset']["category_columns"], "an error occurs in category column(s)."

    ##Check range of air temperature
    assert input_data[config['dataset']["float_columns"][0]].between(
        config['data_defense']["range_air_temperature"]['min_value'], 
        config['data_defense']["range_air_temperature"]['max_value']).sum(
                ) == len_input_data, "an error occurs in air temperature range."
       
    ##Check ranget of process temperature
    assert input_data[config['dataset']["float_columns"][1]].between(
        config['data_defense']["range_process_temperature"]['min_value'], 
        config['data_defense']["range_process_temperature"]['max_value']).sum(
        ) == len_input_data, "an error occurs in process temperature range."

    ##Check range of torque    
    assert input_data[config['dataset']["float_columns"][2]].between(
        config['data_defense']["range_torque"]['min_value'], 
        config['data_defense']["range_torque"]['max_value']).sum(    
        ) == len_input_data, "an error occurs in torque range."
   
    ##Check range of rotational speed
    assert input_data[config['dataset']["int_columns"][0]].between(
        config['data_defense']["range_rotational_speed"]['min_value'], 
        config['data_defense']["range_rotational_speed"]['max_value']).sum(
        ) == len_input_data, "an error occurs in rotational range."
        
    ##Check range of tool wear
    assert input_data[config['dataset']["int_columns"][1]].between(
        config['data_defense']["range_tool_wear"]['min_value'], 
        config['data_defense']["range_tool_wear"]['max_value']).sum(
        ) == len_input_data, "an error occurs in tool wear range."
    
    ##Check range of target    
    assert input_data[config['dataset']["int_columns"][2]][0] in \
        config['data_defense']['target']['value'], \
            "an error occurs in target range."

    ##Check Machine type   
    assert input_data[config['dataset']['category_columns'][0]][0] in \
        config['data_defense']['type']['value'], \
            "an error occurs in type."
   
def split_data(input_data: pd.DataFrame, config: dict):
    # Split predictor and label
    X = input_data[config['dataset']["predictors"]].copy()
    y = input_data[config['dataset']["label"]].copy()

    # 1st split train and test
    X_train, X_test, \
    y_train, y_test = train_test_split(
        X, y,
        test_size = config['dataset']["test_size"],
        random_state = 42,
        stratify = y
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config() 

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config)

    # 3. Drop Columns
    raw_dataset=drop_column(raw_dataset, config)

    # 4. Handling missing data 
    raw_dataset=drop_rows_with_missing_data(raw_dataset, config)

    # 5. Data defense
    check_data(raw_dataset, config)

    # 6. Splitting train, valid, and test set
    X_train, X_test, \
        y_train, y_test = split_data(raw_dataset, config)

    # 7. Save train-test set
    utils.pickle_dump(raw_dataset, '../' + config['train_test']["directory"] + config['train_test']['clean_data'])

    utils.pickle_dump(X_train, '../' + config['train_test']['directory'] + config['train_test']["X_train"])
    utils.pickle_dump(y_train, '../' + config['train_test']['directory'] + config['train_test']["y_train"])

    utils.pickle_dump(X_test, '../' + config['train_test']['directory'] + config['train_test']["X_test"])
    utils.pickle_dump(y_test, '../' + config['train_test']['directory'] + config['train_test']["y_test"])