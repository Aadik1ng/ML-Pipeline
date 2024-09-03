import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('data_visualization')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_visualization.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Ensure the "plots" directory exists
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

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

def visualize_target_distribution(df: pd.DataFrame, stock_names: list) -> None:
    """Visualize the distribution of the target variables."""
    try:
        for stock in stock_names:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[df['Stock_Name'] == stock]['Price'], kde=True, bins=30)
            plt.title(f'Distribution of {stock} Price')
            plt.xlabel('Price')
            plt.ylabel('Frequency')
            plot_path = os.path.join(plots_dir, f'{stock}_price_distribution.png')
            plt.savefig(plot_path)
            logger.debug('Price distribution plot for %s saved to %s', stock, plot_path)
            plt.show()
    except Exception as e:
        logger.error('Error visualizing target distribution: %s', e)
        raise

def scatter_plots(df: pd.DataFrame, stock_names: list) -> None:
    """Analyze relationships between the price and volume for the selected stocks using scatter plots."""
    try:
        for stock in stock_names:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[df['Stock_Name'] == stock]['Volume'], y=df[df['Stock_Name'] == stock]['Price'])
            plt.title(f'{stock} Volume vs Price')
            plt.xlabel('Volume')
            plt.ylabel('Price')
            plot_path = os.path.join(plots_dir, f'{stock}_volume_vs_price.png')
            plt.savefig(plot_path)
            logger.debug('Scatter plot for %s saved to %s', stock, plot_path)
            plt.show()
    except Exception as e:
        logger.error('Error generating scatter plots: %s', e)
        raise

def visualize_correlation_matrix(df: pd.DataFrame, stock_pair: list) -> None:
    """Visualize the correlation between stock prices."""
    try:
        plt.figure(figsize=(8, 6))
        correlation_matrix = df[df['Stock_Name'].isin(stock_pair)].pivot(index='Date', columns='Stock_Name', values='Price').corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f'Correlation between {" and ".join(stock_pair)}')
        plot_path = os.path.join(plots_dir, f'correlation_{"_".join(stock_pair)}.png')
        plt.savefig(plot_path)
        logger.debug('Correlation matrix plot for %s saved to %s', " and ".join(stock_pair), plot_path)
        plt.show()
    except Exception as e:
        logger.error('Error visualizing correlation matrix: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        file_path = params['data_visualization']['file_path']
        df = load_data(file_path)
        
        # Visualize distributions and scatter plots
        stock_names = params['data_visualization']['stock_names']
        visualize_target_distribution(df, stock_names)
        scatter_plots(df, stock_names)
        
        # Visualize correlations
        correlation_pairs = params['data_visualization']['correlation_pairs']
        for stock_pair in correlation_pairs:
            visualize_correlation_matrix(df, stock_pair)
        
    except Exception as e:
        logger.error('Failed to complete the data visualization process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
