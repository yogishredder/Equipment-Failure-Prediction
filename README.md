
# Equipment Failure Prediction (End-to-end Machine Learning Process)
## Documentation

    1. Documentation of API (Input)
![Documentation of API (Input)](https://github.com/yogishredder/Equipment-Failure-Prediction/blob/main/config/assets/Documentation_Input%20API.png)

    2. Documentation of API (Output)
![Documentation of API (Output)](https://github.com/yogishredder/Equipment-Failure-Prediction/blob/main/config/assets/Documentation_Output_API.png)

    3. Documentation of Streamlit
![Documentation of Streamlit (Input)](https://github.com/yogishredder/Equipment-Failure-Prediction/blob/main/config/assets/Streamlit.png)

![Documentation of Streamlit (Output)](https://github.com/yogishredder/Equipment-Failure-Prediction/blob/main/config/assets/Streamlit_Output.png)

    4. Documentation of Workflow (API) 

![Documentation of workflow_API)](https://github.com/yogishredder/Equipment-Failure-Prediction/blob/main/config/assets/Documentation_Workflow.png)

    5. Documentation of Workflow (Streamlit) 

![Documentation of workflow_API)](https://github.com/yogishredder/Equipment-Failure-Prediction/blob/main/config/assets/Documentation_Workflow_Streamlit.png)

## Business Problem & Objective

Maintenance refers to the process of ensuring that equipment, machinery, facilities, or any other assets operate at their intended level of performance. It involves various tasks, inspections, repairs, and upkeep activities aimed at preventing equipment failures, extending their lifespan, and maximizing their efficiency and reliability.

In plants, sensors and instruments are installed to monitor the parameters of equipment. Those parameters can be an indication to determine whether machinery equipment is in good or bad condition. It is important to take action in order to mitigate the equipment before equipment failures become catastrophes. It can make a company lose its efficiency and suffer higher maintenance costs.
Objective

In this project, a prediction approach (Classification) is developed to mitigate equipment failures based on its operational parameters to enhance overall equipment reliability and safety.

## Project Architecture

![Project Architecture](https://github.com/yogishredder/Equipment-Failure-Prediction/blob/main/config/assets/Flowchart%20project.png)
## How to Use

To use the prediction, the following steps are needed:
1. Create a virtual environment 

```bash
  python3 -m venv
```

2. Activate the virtual environment

```bash
  source venv/bin/activate
```

3. Install required libraries in requirements.txt

```bash
  pip install -r requirements.txt
```
4. Run data preparation

```bash
  python3 data_preparation.py
```

5. Run data preprocessing

```bash
  python3 data_preprocessing.py
```

6. Run model

```bash
  python3 model.py
```

7. Run API

```bash
  python3 api.py then
  open http://localhost:8000/docs in a browser
```

7. Run streamlit

```bash
  streamlit run streamlit.py then
  CTRL + Click the given URL in a browser
```

8. Insert parameter value in the streamlit dashboard

9. Click predict

The parameters used for prediction are:

    1. Type
    2. Air Temperature [K]
    3. Process temperature [K]
    4. Rotational speed [rpm]
    5. Torque [Nm]
    6. Tool wear [min]

The outcome of the prediction is either:
    
    1. Machine is normal, or
    2. Machine is broken
## Dataset
The dataset used in the project is AI4I 2020 Predictive Maintenance Dataset, which is a synthetic dataset that reflects real predictive maintenance data encountered in industry. This is the link to the dataset https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset.

The columns that are used in the project are Type, Air Temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], and Tool wear [min].
## Conclusion

Conclusion
After all those processes, concluded:

    1. The final model is DecisionTreeClassifier with parameters criterion max_depth = 5
    2. Based on the model, the accuracy of the model is 90%
## Future Work

Develop a model with more data and a more advanced model (deep learning) that can classify equipment failure more perfectly
## Reference

    1. Pacmann: https://pacmann.io/

    2. UC Irvine Repository: https://archive.ics.uci.edu/datasets

    3. Stanford Project Report: http://cs230.stanford.edu/past-projects/ 
## Authors

- [@yogishredder](https://github.com/yogishredder)
