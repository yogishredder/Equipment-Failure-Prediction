import util as utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def load_dataset_feng(config_data: dict):
    # Load every set of data
    X_train_feng = utils.pickle_load('../' + config['train_test']['directory'] + config['train_test']["X_train_feng"])
    y_train_feng = utils.pickle_load('../' + config['train_test']['directory'] + config['train_test']["y_train_feng"])

    X_test_feng = utils.pickle_load('../' + config['train_test']['directory'] + config['train_test']["X_test_feng"])
    y_test_feng = utils.pickle_load('../' + config['train_test']['directory'] + config['train_test']["y_test_feng"])

    # Return 3 set of data
    return X_train_feng, y_train_feng, X_test_feng, y_test_feng

def train_model(X_train_feng, y_train_feng, X_test_feng, y_test_feng):
    param = config['best_model']['parameter']
    model = DecisionTreeClassifier(**param)
    model.fit(X_train_feng, y_train_feng)
    y_pred = model.predict(X_test_feng)
    print(classification_report(y_test_feng, y_pred))

    return model

if __name__ == "__main__" :
    # 1. Load config file
    config = utils.load_config()

    # 2. Load set data
    X_train_feng, y_train_feng, X_test_feng, y_test_feng = load_dataset_feng(config)

    # 3. Train model
    model = train_model(X_train_feng, y_train_feng, X_test_feng, y_test_feng)

    # 4. Dump model
    utils.pickle_dump(model, '../' + config['best_model']['model_path'] + config['best_model']['model_name'])