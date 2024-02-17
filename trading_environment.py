import pandas as pd
import torch
from collections import defaultdict


class TradingEnvironment:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.features = preprocess_data(self.data)
        self.balance = 100_000.00
        self.profit = 0.00
        self.drawdown = 0.00
        self.shares_owned = 0
        self.balances = defaultdict(list)

        
    def simulate_trading(self):
        self.profit = 0.00
        max_profit = 0
        self.drawdown = 0
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

    # Drop columns
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
    
    # Fill NaN values
    data_filled = data.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # Convert the DataFrame to a tensor
    features_tensor = torch.tensor(data_filled.values, dtype=torch.float32)
    
    return features_tensor

