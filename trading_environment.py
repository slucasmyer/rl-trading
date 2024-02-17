import pandas as pd
import torch
from collections import defaultdict


class TradingEnvironment:
    """
    A class to simulate trading in a stock market.
    The environment is initialized with a dataset and a model.
    The model is used to make trading decisions based on the dataset.
    Profit and drawdown are calculated based on the trading decisions.
    """
    def __init__(self, data, model):
        self.data = data # The dataset
        self.model = model # The model
        self.features = preprocess_data(self.data) # Preprocessed data
        self.balance = 100_000.00 # Initial balance
        self.profit = 0.00 # Profit
        self.drawdown = 0.00 # Drawdown
        self.shares_owned = 0 # Shares owned
        self.balances = defaultdict(list) # Balances over time

        
    def simulate_trading(self):
        """
        Currently the core of this class.
        Resets profit and drawdown for evaluation of new individual.
        Simulates trading over the dataset.
        Updates max_profit and drawdown.
        Returned values are used to evaluate the individual (multi-objective).
        """
        self.profit = 0.00 # Reset profit for evaluation of new individual
        max_profit = 0 # Max profit
        self.drawdown = 0 # Reset drawdown for evaluation of new individual
        # Simulate trading over the dataset
        for i in range(len(self.features)):
          
            feature_vector = self.features[i:i+1] # Get the feature vector for the current day

            feature_vector = feature_vector.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            decision = self.model(feature_vector).argmax().item()  # 0=buy, 1=hold, 2=sell

            current_price = self.data['close'].iloc[i]

            #   if decision == 0:  # Buy
            #       self.profit -= self.data['close'][i]
            #   elif decision == 2:  # Sell
            #       self.profit += self.data['close'][i]
            if decision == 0 and self.balance >= current_price:  # Buy
                self.shares_owned += self.balance // current_price
                self.balance -= current_price * self.shares_owned
                
            elif decision == 2 and self.shares_owned > 0:  # Sell
                self.balance += current_price * self.shares_owned
                self.shares_owned = 0
          
        # Update max_profit and drawdown
        max_profit = max(max_profit, self.profit)
        self.drawdown = min(self.drawdown, self.profit - max_profit)

        return self.profit, self.drawdown


def preprocess_data(data, columns_to_drop=[]):
    """
    This will likely be unnecessary soon.
    Can be moved to data_preparation.py.
    All we truly need here is to convert the DataFrame to a tensor.
    May be useful to implement column dropping optionality there too.
    """
    # Drop columns
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
    
    # Fill NaN values
    data_filled = data.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # Convert the DataFrame to a tensor
    features_tensor = torch.tensor(data_filled.values, dtype=torch.float32)
    
    return features_tensor

