import pandas as pd
import os
import logging
import yaml
import joblib
from sklearn.preprocessing import LabelEncoder

# Ensure the "logs" and "models" directories exist
log_dir = 'logs'
models_dir = 'models'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
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

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    try:
        # Convert numeric columns to proper types
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            df[col] = df[col].str.replace(',', '').astype(float, errors='ignore')
        
        # Fill missing values with mean for numeric columns
        df = df.fillna(df.mean(numeric_only=True))
        logger.debug('Missing values filled with mean')
        
        return df
    except Exception as e:
        logger.error('Error handling missing values: %s', e)
        raise

def convert_stock_names_to_ids(df: pd.DataFrame, column: str) -> tuple:
    """Convert stock names to unique IDs and return the encoder."""
    try:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        logger.debug('Converted stock names to unique IDs')
        return df, le  # Return both DataFrame and LabelEncoder
    except Exception as e:
        logger.error('Error converting stock names to unique IDs: %s', e)
        raise

def create_time_based_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Create time-based features from a timestamp column."""
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['year'] = df[timestamp_col].dt.year
        df['month'] = df[timestamp_col].dt.month
        df['day'] = df[timestamp_col].dt.day
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['is_weekend'] = df[timestamp_col].dt.dayofweek >= 5
        logger.debug('Time-based features created')
        return df
    except Exception as e:
        logger.error('Error creating time-based features: %s', e)
        raise

def create_lag_features(df: pd.DataFrame, price_columns: list, lags: list) -> pd.DataFrame:
    """Create lag features for specified columns."""
    try:
        for col in price_columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                logger.debug('Created lag feature for %s with lag %d', col, lag)
        return df
    except Exception as e:
        logger.error('Error creating lag features: %s', e)
        raise

def create_rolling_statistics(df: pd.DataFrame, price_columns: list, windows: list) -> pd.DataFrame:
    """Create rolling statistics (mean and standard deviation) for specified columns."""
    try:
        for col in price_columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
        logger.debug('Rolling statistics creation completed')
        return df
    except Exception as e:
        logger.error('Error creating rolling statistics: %s', e)
        raise

def create_features(df: pd.DataFrame, lags: list, windows: list) -> pd.DataFrame:
    """Generate new features from the existing dataset."""
    try:
        price_columns = [col for col in df.columns if 'Price' in col]

        # Convert categorical stock names to unique IDs
        stock_columns = [col for col in df.columns if 'Stock' in col]  # Adjust column name if needed
        for col in stock_columns:
            df, _ = convert_stock_names_to_ids(df, col)  # Ensure to handle each stock column

        # Time-based features
        if 'Date' in df.columns:
            df = create_time_based_features(df, timestamp_col='Date')

        # Lag features
        df = create_lag_features(df, price_columns, lags)

        # Rolling statistics
        df = create_rolling_statistics(df, price_columns, windows)

        logger.debug('Feature engineering completed')
        return df
    except Exception as e:
        logger.error('Error during feature engineering: %s', e)
        raise

def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the engineered dataset to a CSV file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.debug('Data saved to %s', output_path)
    except Exception as e:
        logger.error('Error saving data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        file_path = params['feature_engineering']['file_path']  # Path to the raw data CSV
        output_path = params['feature_engineering']['output_path']  # Path to save the engineered data
        
        df = load_data(file_path)
        df = handle_missing_values(df)
        lags = params['feature_engineering']['lags']
        windows = params['feature_engineering']['windows']
        
        # Example of converting stock names in a specific column
        df, le = convert_stock_names_to_ids(df, 'Stock_Name')
        df = create_features(df, lags, windows)
        save_data(df, output_path)
        
        # Save the encoder to a file
        joblib.dump(le, os.path.join(models_dir, 'label_encoder.pkl'))
        logger.debug('Label encoder saved to label_encoder.pkl')
        
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
