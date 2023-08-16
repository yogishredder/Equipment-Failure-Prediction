import streamlit as st
import requests
import joblib
from PIL import Image
from util import *
import pandas as pd
import numpy as np

config = load_config()

file_name = '../' + config['dataset']['data_directory'] + config['dataset']['file_name']
dataset = pd.read_csv(file_name)

# function to return categories in columns
def label_streamlit(dataset, columns):
    label = dataset[columns].unique().tolist()
    #label = [value for value in label if str(value) != '?']
    #label = [value for value in label if str(value) != 'nan']
    #label = [value for value in label if str(value) != 'Other']

    return np.sort(label).tolist()

# Load and set images in the first place
header_images = Image.open('../'+'config/assets/Mill Machine.png')
st.image(header_images)

# Add some information about the service
st.title("Equipment Failure Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "Parameter Input"):
    # Create box for number input

    type = st.selectbox(
        label = 'Type',
        options = [''] + label_streamlit(dataset = dataset, columns = 'Type')
    )

    air_temperature = st.number_input(
        label = 'Air temperature [K]',
        min_value = 295.3,
        max_value = 304.5,
        help = "Value range from 295.3 to 304.5"
    )

    process_temperature = st.number_input(
        label = 'Process temperature [K]',
        min_value = 305.7,
        max_value = 313.8,
        help = "Value range from 305.7 to 313.8"
    )

    rotational_speed = st.number_input(
        label = "Rotational speed [rpm]",
        min_value = 1168,
        max_value = 2886,
        help = "Value range from 1168 to 2886"
    )

    torque = st.number_input(
        label = "Torque [Nm]",
        min_value = 3.8,
        max_value = 76.6,
        help = "Value range from 3.8 to 76.6"
    )

    tool_wear = st.number_input(
        label = "Tool wear [min]",
        min_value = 0,
        max_value = 253,
        help = "Value range from 0 to 253"
    )

    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        form_data = {
            "Type": type,
            "Air temperature [K]": air_temperature,
            'Process temperature [K]': process_temperature,
            'Rotational speed [rpm]': rotational_speed,
            'Torque [Nm]': torque,
            "Tool wear [min]": tool_wear,
        }

        # Create loading animation while predicting
        # sending data to api
        with st.spinner('Sending data to prediction server ...'):
            res = requests.post('http://localhost:8000/predict/', json = form_data).json()
        #st.write(res)
        
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Machine is normal.":
                st.warning("Machine is broken.")
            else:
                st.success("Machine is normal.")