import pandas as pd
import os
import logging
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib  # For saving the model
import mlflow
import mlflow.sklearn
import dagshub
# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

dagshub.init(repo_owner='Aadik1ng', repo_name='HubbleMind', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Aadik1ng/HubbleMind.mlflow")

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise

def get_model(model_name: str, hyperparams: dict):
    """Retrieve the machine learning model based on the model name and hyperparameters."""
    mlflow.log_param("model_type", "RandomForest")

    try:
        if model_name == 'LinearRegression':
            return LinearRegression(**hyperparams)
        elif model_name == 'DecisionTreeRegressor':
            return DecisionTreeRegressor(**hyperparams)
        elif model_name == 'RandomForestRegressor':
            return RandomForestRegressor(**hyperparams)
        elif model_name == 'SVR':
            return SVR(**hyperparams)
        elif model_name == 'MLPRegressor':
            return MLPRegressor(**hyperparams)
        else:
            logger.error('Unsupported model type: %s', model_name)
            raise ValueError('Unsupported model type')
    except Exception as e:
        logger.error('Error retrieving the model: %s', e)
        raise

def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Train the specified machine learning model."""

    try:
        model.fit(X_train, y_train)

        logger.debug('Model trained successfully')
        return model
    except Exception as e:
        logger.error('Error training the model: %s', e)
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate the model and return the mean squared error."""
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.debug('Model evaluation completed with MSE: %f', mse)
        return mse
    except Exception as e:
        logger.error('Error evaluating the model: %s', e)
        raise

def save_model(model, model_path: str) -> None:
    """Save the trained model to a file."""
    try:
        joblib.dump(model, model_path)
        logger.debug('Model saved to %s', model_path)
    except Exception as e:
        logger.error('Error saving the model: %s', e)
        raise
def save_data(data, path, filename):
    """Save data to a file in the specified path."""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, filename)
        data.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving data: %s', e)
        raise

def main():
    try:
        mlflow.start_run()

        # Load parameters
        params = load_params(params_path='params.yaml')
        model_dir='models'
        os.makedirs(model_dir, exist_ok=True)

        data_file = params['model_building']['data_file']
        model_name = params['model_building']['model_name']
        hyperparams = params.get('model_building', {}).get('hyperparameters', {})
        model_path = params['model_building']['model_path']
        test_size = params['model_building'].get('test_size', 0.2)
        save_path=params['model_building']['save_path']
        # Load data
        df = load_data(file_path=data_file)
        X = df.drop(columns=['Price','Date'])  # Features
        y = df['Price']  # Target

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        save_data(X_train, save_path, 'X_train.csv')
        save_data(X_test, save_path, 'X_test.csv')
        save_data(y_train.to_frame(), save_path, 'y_train.csv')
        save_data(y_test.to_frame(), save_path, 'y_test.csv')

        # Get and train the model
        model = get_model(model_name, hyperparams)
        for param, value in hyperparams.items():
            mlflow.log_param(param, value)
        model = train_model(model, X_train, y_train)
        mlflow.sklearn.log_model(model, "model")

        # Evaluate the model
        mse = evaluate_model(model, X_test, y_test)
        logger.info('Model performance: Mean Squared Error (MSE) = %.4f', mse)
        mlflow.log_metric("mse", mse)
        # Save the model
        save_model(model, model_path)
        logger.info('Model saved to %s', model_path)
        mlflow.end_run()

        logger.info('Model training and saving process completed successfully')
    except Exception as e:
        logger.error('Failed to complete the model training process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
