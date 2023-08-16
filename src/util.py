import yaml
import joblib

config_dir = "config/config.yaml"

def load_config() -> dict: 
    # Try to load yaml file
    try:
        with open(config_dir, "r") as file:
            config = yaml.safe_load(file)
    except:
        config = yaml.safe_load(open('../' + config_dir))

    # Return params in dict format
    return config

def pickle_dump(data, file_path):
    """
    Serialize and save Python object to a joblib pickle file.

    Parameters:
        data: Any - The Python object to be serialized.
        file_path (str): The path of the pickle file to save the data.

    Returns:
        None
    """
    joblib.dump(data, file_path)

def pickle_load(file_path):
    """
    Load and deserialize Python object from a joblib pickle file.

    Parameters:
        file_path (str): The path of the pickle file to load the data.

    Returns:
        Any: The Python object loaded from the pickle file.
    """
    data = joblib.load(file_path)
    return data