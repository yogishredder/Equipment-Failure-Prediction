import data_preprocessing
import util as utils
import pandas as pd
import numpy as np
import numpy as np

def test_label_encoder_data():
    #arrange
    config = utils.load_config()

    mock_data = ['M', 'M', 'M', 'L', 'H']
    
    mock_data = pd.DataFrame(mock_data, columns=['Type'])

    expected_data = [2, 2, 2, 1, 3] 
    
    expected_data = pd.DataFrame(mock_data, columns=['Type'])

    #act
    processed_data = data_preprocessing.encoding_category(mock_data, config) 

    #assert
    assert processed_data.equals(expected_data)
