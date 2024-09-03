import pandas as pd
import yaml
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import logging
import os

# Logging configuration
logger = logging.getLogger('stock_price_prediction')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/stock_price_prediction.log')
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
    except Exception as e:
        logger.error('Error loading params: %s', e)
        raise

def load_model(model_path: str):
    """Load the trained model from a file."""
    try:
        model = joblib.load(model_path)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model: %s', e)
        raise

def load_label_encoder(encoder_path: str) -> LabelEncoder:
    """Load the label encoder from a file."""
    try:
        le = joblib.load(encoder_path)
        logger.debug('Label encoder loaded from %s', encoder_path)
        return le
    except Exception as e:
        logger.error('Error loading label encoder: %s', e)
        raise

def create_feature_data(stock_name: str, days: int, X_test: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    """Create future feature data for prediction."""
    try:
        # Ensure the necessary columns are present
        required_columns = [
            'Volume', 'index', 'Stock_Name', 'year', 'month', 'day', 
            'day_of_week', 'is_weekend', 'Price_lag_1', 'Price_lag_2', 
            'Price_lag_3', 'Price_rolling_mean_5', 'Price_rolling_std_5', 
            'Price_rolling_mean_10', 'Price_rolling_std_10'
        ]
        
        for col in required_columns:
            if col not in X_test.columns:
                raise ValueError(f"Column '{col}' is missing from X_test")
        
        # Convert stock name to ID
        stock_id = le.transform([stock_name])[0]

        # Filter data for the specific stock
        stock_data = X_test[X_test['Stock_Name'] == stock_id]
        
        if stock_data.empty:
            raise ValueError(f"No data found for stock name {stock_name}")

        # Get the latest row of the stock data
        latest_row = stock_data.sort_values(by='index', ascending=False).iloc[0]
        
        # Create future dates
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days + 1)]
        
        # Initialize future_data with the required columns
        future_data = pd.DataFrame(columns=required_columns)
        
        # Fill the future_data DataFrame
        for future_date in future_dates:
            new_row = latest_row.copy()
            new_row['Stock_Name'] = stock_id
            new_row['year'] = future_date.year
            new_row['month'] = future_date.month
            new_row['day'] = future_date.day
            new_row['day_of_week'] = future_date.weekday()
            new_row['is_weekend'] = int(future_date.weekday() >= 5)  # 1 for weekend, 0 for weekday
            
            # Use last known values for price-related features
            future_data = pd.concat([future_data, pd.DataFrame([new_row[required_columns]], columns=required_columns)], ignore_index=True)
        
        logger.debug('Feature data for prediction created successfully')
        logger.debug('Future data:\n%s', future_data.head())
        return future_data
    except Exception as e:
        logger.error('Error creating feature data: %s', e)
        raise

def predict_stock_prices(model, future_data: pd.DataFrame) -> pd.DataFrame:
    """Predict future stock prices."""
    try:
        logger.debug('Data for prediction:\n%s', future_data.head())
        predictions = model.predict(future_data)
        future_data['Predicted_Price'] = predictions
        return future_data[['Predicted_Price']]
    except Exception as e:
        logger.error('Error during prediction: %s', e)
        raise

def save_predictions(predictions: pd.DataFrame, stock_name: str, output_dir: str):
    """Save predictions to a CSV file."""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f'{stock_name}_predictions.csv')
        predictions.to_csv(output_file, index=False)
        logger.info('Predictions saved to %s', output_file)
    except Exception as e:
        logger.error('Error saving predictions: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')
        model = load_model(params['prediction']['model_path'])
        le = load_label_encoder('./models/label_encoder.pkl')

        stock_name = params['prediction']['stock_name']
        days = params['prediction']['prediction_period']  # Number of days to predict
        
        # Load X_test to create future feature data
        X_test = pd.read_csv(params['prediction']['X_test_path'])

        future_data = create_feature_data(stock_name, days, X_test, le)
        predictions = predict_stock_prices(model, future_data)
        save_predictions(predictions, stock_name, 'predictions')
        
    except Exception as e:
        logger.error('Failed to complete the prediction process: %s', e)

if __name__ == '__main__':
    main()
