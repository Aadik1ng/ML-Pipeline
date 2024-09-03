import pandas as pd
import os
import logging
import yaml
import joblib  # For loading the model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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

def load_model(model_path: str):
    """Load the trained model from a file."""
    try:
        model = joblib.load(model_path)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading the model: %s', e)
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate the model using test data."""
    try:
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info('Model Evaluation Metrics:')
        logger.info(f'Mean Squared Error (MSE): {mse}')
        logger.info(f'Mean Absolute Error (MAE): {mae}')
        logger.info(f'R^2 Score: {r2}')
        
        return mse, mae, r2, predictions
    except Exception as e:
        logger.error('Error evaluating the model: %s', e)
        raise

def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5):
    """Perform cross-validation and calculate mean and standard deviation of scores."""
    try:
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        logger.info('Cross-Validation Metrics:')
        logger.info(f'Mean Cross-Validation Score: {-mean_score}')
        logger.info(f'Standard Deviation of Cross-Validation Scores: {std_score}')
        
        return -mean_score, std_score
    except Exception as e:
        logger.error('Error during cross-validation: %s', e)
        raise

def test_on_unseen_data(model, X_test: pd.DataFrame, y_test: pd.Series, recent_data_ratio: float):
    """Test the model on recent data from the test set."""
    try:
        split_idx = int(len(X_test) * (1 - recent_data_ratio))
        X_recent = X_test.iloc[split_idx:]
        y_recent = y_test.iloc[split_idx:]
        
        predictions_recent = model.predict(X_recent)
        mse_recent = mean_squared_error(y_recent, predictions_recent)
        mae_recent = mean_absolute_error(y_recent, predictions_recent)
        
        logger.info('Recent Data Evaluation Metrics:')
        logger.info(f'Mean Squared Error (MSE) on Recent Data: {mse_recent}')
        logger.info(f'Mean Absolute Error (MAE) on Recent Data: {mae_recent}')
        
        return mse_recent, mae_recent
    except Exception as e:
        logger.error('Error testing on recent data: %s', e)
        raise

def feature_importance_analysis(model, X_train: pd.DataFrame):
    """Analyze and visualize feature importance for the model."""
    try:
        if hasattr(model, 'coef_'):  # Linear models
            importance = model.coef_
            feature_names = X_train.columns
        elif hasattr(model, 'feature_importances_'):  # Tree-based models
            importance = model.feature_importances_
            feature_names = X_train.columns
        else:
            logger.error('Model does not support feature importance analysis.')
            raise ValueError('Model does not support feature importance analysis.')

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('./plots/feature_importance.png')
        plt.show()
        
        logger.info('Feature importance analysis completed and plotted.')
        return importance_df
    except Exception as e:
        logger.error('Error during feature importance analysis: %s', e)
        raise

def plot_performance_metrics(y_test, predictions):
    """Plot model performance metrics."""
    try:
        plt.figure(figsize=(14, 7))
        
        # Plot actual vs. predicted values
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')

        # Plot residuals
        residuals = y_test - predictions
        plt.subplot(1, 2, 2)
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')

        plt.tight_layout()
        plt.savefig('./plots/model_performance.png')
        plt.show()

        logger.info('Performance metrics plotted.')
    except Exception as e:
        logger.error('Error plotting performance metrics: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        X_test = pd.read_csv(params['model_evaluation']['X_test_path'])
        y_test = pd.read_csv(params['model_evaluation']['y_test_path']).squeeze()  # Convert DataFrame to Series
        X_train = pd.read_csv(params['model_evaluation']['X_train_path'])
        y_train = pd.read_csv(params['model_evaluation']['y_train_path']).squeeze()  # Convert DataFrame to Series
        model_path = params['model_evaluation']['model_path']  # Path to the trained model
        
        model = load_model(model_path)
        mse, mae, r2, predictions = evaluate_model(model, X_test, y_test)
        mean_cv_score, std_cv_score = cross_validate_model(model, X_train, y_train)
        mse_recent, mae_recent = test_on_unseen_data(model, X_test, y_test, recent_data_ratio=0.2)
        importance_df = feature_importance_analysis(model, X_train)
        plot_performance_metrics(y_test, predictions)
        
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
