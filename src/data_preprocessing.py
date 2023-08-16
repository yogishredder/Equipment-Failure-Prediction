import pandas as pd
import util as utils
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
import numpy as np

config = utils.load_config()
def load_dataset(config_data: dict):
    # Load every set of data
    X_train = utils.pickle_load('../' + config['train_test']['directory'] + config['train_test']["X_train"])
    y_train = utils.pickle_load('../' + config['train_test']['directory'] + config['train_test']["y_train"])

    X_test = utils.pickle_load('../' + config['train_test']['directory'] + config['train_test']["X_test"])
    y_test = utils.pickle_load('../' + config['train_test']['directory'] + config['train_test']["y_test"])

    # Concatenate x and y each set
    train_set = pd.concat([X_train, y_train], axis = 1)
    test_set = pd.concat([X_test, y_test], axis = 1)

    # Return 3 set of data
    return train_set, test_set

def rus_fit_resample(set_data, config):
    X_rus, y_rus = RandomUnderSampler(random_state = 42).fit_resample(
        train_set.drop(columns = config['dataset']["label"]), 
        train_set[config['dataset']["label"]]
        )
    train_set_bal = pd.concat([X_rus, y_rus], axis = 1)

    return train_set_bal

def splittypedata(data):
    data_float = data[config['dataset']["float_columns"]]
    data_int = data[config['dataset']["int_columns"]]
    data_category = data[config['dataset']["category_columns"]]

    return data_float, data_int, data_category

def remove_outliers(set_data):
    set_data = set_data.copy()
    list_of_set_data = list()

    for col_name in set_data.columns[:-1]:
        q1 = set_data[col_name].quantile(0.25)
        q3 = set_data[col_name].quantile(0.75)
        iqr = q3 - q1
        set_data_cleaned = set_data[~((set_data[col_name] < (q1 - 1.5 * iqr)) | (set_data[col_name] > (q3 + 1.5 * iqr)))].copy()
        list_of_set_data.append(set_data_cleaned.copy())
    
    set_data_cleaned = pd.concat(list_of_set_data)
    count_duplicated_index = set_data_cleaned.index.value_counts()
    used_index_data = count_duplicated_index[count_duplicated_index == (set_data.shape[1]-1)].index
    set_data_cleaned = set_data_cleaned.loc[used_index_data].drop_duplicates()

    return set_data_cleaned

def encoding_category(data):
    # Make a copy of the DataFrame to avoid inplace modification
    encoded_data = data.copy()

    # Replace values in the specified 'category_columns'
    category_columns = config['dataset']['category_columns']
    encoded_data[category_columns] = encoded_data[category_columns].replace({'L': 1, 'M': 2, 'H': 3})

    return encoded_data

def concatcleandata(set_data_cleaned):
    count_duplicated_index = set_data_cleaned.index.value_counts()
    used_index_data = count_duplicated_index[count_duplicated_index == 1].index
    set_data_cleaned = set_data_cleaned.loc[used_index_data].drop_duplicates()
    set_data_cleaned = set_data_cleaned.dropna()

    return set_data_cleaned

if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Load dataset
    train_set, test_set = load_dataset(config)

    # 3. Undersampling dataset
    train_set_bal = rus_fit_resample(train_set, config)

    # 4. Split data type
    train_set_bal_float, train_set_bal_int, train_set_bal_category = splittypedata(train_set_bal)
    
    # 5. Removing outliers
    train_set_bal_cleaned_float = remove_outliers(train_set_bal_float)
    train_set_bal_cleaned_int = remove_outliers(train_set_bal_int)

    # 6. Label Encoder
    #train_set_bal_cat_encoder = labelencoder(train_set_bal_category)
    train_set_bal_cat_encoder = encoding_category(train_set_bal_category)  

    # 7. Concat label encoder
    train_set_bal_cleaned_concat = pd.concat([train_set_bal_cat_encoder, 
                                          train_set_bal_cleaned_float, 
                                          train_set_bal_cleaned_int 
                                          ],
                                        axis=1
                                        )

    train_set_bal_cleaned_concat = concatcleandata(train_set_bal_cleaned_concat)
   
    # 8. Test
    test_float, test_int, test_category = splittypedata(test_set)
   
    test_category_encoder = encoding_category(test_category)
    test_set_encoder_concat = pd.concat([test_category_encoder, test_float, test_int],
                                     axis=1
                                    )
    test_set_encoder_concat = concatcleandata(test_set_encoder_concat)
    
    # 9. Dump set data
    #Dump Train data (feature engineering)
    utils.pickle_dump(train_set_bal_cleaned_concat[config['dataset']["predictors"]], 
                      '../' + config['train_test']['directory'] + config['train_test']['X_train_feng'])
    utils.pickle_dump(train_set_bal_cleaned_concat[config['dataset']["label"]], 
                      '../' + config['train_test']['directory'] + config['train_test']['y_train_feng'])
    #Dump test data (feature engineering)
    utils.pickle_dump(test_set_encoder_concat[config['dataset']["predictors"]], 
                      '../' + config['train_test']['directory'] + config['train_test']['X_test_feng'])
    utils.pickle_dump(test_set_encoder_concat[config['dataset']["label"]], 
                      '../' + config['train_test']['directory'] + config['train_test']['y_test_feng'])
    
    #print(train_set_bal_cat_encoder)
    #print(train_set_bal_category)
   