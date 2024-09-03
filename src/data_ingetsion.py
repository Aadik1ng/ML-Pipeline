import pandas as pd
import os
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Expected columns
expected_columns = ['Unnamed: 0', 'Date', 'Natural_Gas_Price', 'Natural_Gas_Vol.', 'Crude_oil_Price', 'Crude_oil_Vol.', 'Copper_Price', 'Copper_Vol.', 'Bitcoin_Price', 'Bitcoin_Vol.', 'Platinum_Price', 'Platinum_Vol.', 'Ethereum_Price', 'Ethereum_Vol.', 'S&P_500_Price', 'Nasdaq_100_Price', 'Nasdaq_100_Vol.', 'Apple_Price', 'Apple_Vol.', 'Tesla_Price', 'Tesla_Vol.', 'Microsoft_Price', 'Microsoft_Vol.', 'Silver_Price', 'Silver_Vol.', 'Google_Price', 'Google_Vol.', 'Nvidia_Price', 'Nvidia_Vol.', 'Berkshire_Price', 'Berkshire_Vol.', 'Netflix_Price', 'Netflix_Vol.', 'Amazon_Price', 'Amazon_Vol.', 'Meta_Price', 'Meta_Vol.', 'Gold_Price', 'Gold_Vol.']

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

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file and check column name consistency."""
    try:
        df = pd.read_csv(data_url)
        if list(df.columns) == expected_columns:
            logger.debug('Data loaded from %s', data_url)
            return df
        else:
            logger.error('Column names do not match expected columns.')
            logger.error('Expected columns: %s', expected_columns)
            logger.error('Actual columns: %s', list(df.columns))
            raise ValueError('Column names are inconsistent.')
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform data into a long format with columns: index, date, stock name, price, volumes."""
    try:
        # List of stock columns
        stock_columns = {
            'Natural_Gas': ['Natural_Gas_Price', 'Natural_Gas_Vol.'],
            'Crude_oil': ['Crude_oil_Price', 'Crude_oil_Vol.'],
            'Copper': ['Copper_Price', 'Copper_Vol.'],
            'Bitcoin': ['Bitcoin_Price', 'Bitcoin_Vol.'],
            'Platinum': ['Platinum_Price', 'Platinum_Vol.'],
            'Ethereum': ['Ethereum_Price', 'Ethereum_Vol.'],
            'S&P_500': ['S&P_500_Price', 'Nasdaq_100_Price'],
            'Nasdaq_100': ['Nasdaq_100_Price', 'Nasdaq_100_Vol.'],
            'Apple': ['Apple_Price', 'Apple_Vol.'],
            'Tesla': ['Tesla_Price', 'Tesla_Vol.'],
            'Microsoft': ['Microsoft_Price', 'Microsoft_Vol.'],
            'Silver': ['Silver_Price', 'Silver_Vol.'],
            'Google': ['Google_Price', 'Google_Vol.'],
            'Nvidia': ['Nvidia_Price', 'Nvidia_Vol.'],
            'Berkshire': ['Berkshire_Price', 'Berkshire_Vol.'],
            'Netflix': ['Netflix_Price', 'Netflix_Vol.'],
            'Amazon': ['Amazon_Price', 'Amazon_Vol.'],
            'Meta': ['Meta_Price', 'Meta_Vol.'],
            'Gold': ['Gold_Price', 'Gold_Vol.']
        }
        
        rows = []
        index = 0
        
        for stock_name, (price_col, vol_col) in stock_columns.items():
            if price_col in df.columns and vol_col in df.columns:
                # Reshape each stock's data
                stock_data = df[['Date', price_col, vol_col]].copy()
                stock_data['index'] = index
                stock_data['Stock_Name'] = stock_name
                stock_data.rename(columns={price_col: 'Price', vol_col: 'Volume'}, inplace=True)
                rows.append(stock_data)
                index += 1
        
        transformed_df = pd.concat(rows, ignore_index=True)
        
        # Handle various date formats by using dayfirst=True
        transformed_df['Date'] = pd.to_datetime(transformed_df['Date'], dayfirst=True)
        
        transformed_df.sort_values(by='Date', inplace=True)  # Sort by 'Date'
        logger.debug('Data transformation and sorting completed successfully.')
        return transformed_df
    except Exception as e:
        logger.error('Error during data transformation: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        data_url = params['data_ingestion']['data_url']
        df = load_data(data_url=data_url)
        transformed_df = transform_data(df)
        transformed_df.to_csv('./data/transformed_data.csv', index=False)
        logger.info('Transformed data saved as transformed_data.csv')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
