import pandas as pd
import torch
from time import sleep
from collections import defaultdict


class TradingEnvironment:
    """
    A class to simulate trading in a stock market.
    The environment is initialized with a dataset and a model.
    The model is used to make trading decisions based on the dataset.
    Profit and drawdown are calculated based on the trading decisions.
    """
    def __init__(self, features, model, closing_prices):
        self.features = features # The dataset
        self.model = model # The model
        self.closing_prices = closing_prices

        self.balance = 100_000.00 # Initial balance
        self.profit = 0.00 # Profit
        self.max_profit = 0.00 # Max profit
        self.drawdown = 0.00 # Drawdown
        self.shares_owned = 0 # Shares owned
        self.balances = defaultdict(list) # Balances over time
        self.stop_loss_triggered = defaultdict(list) # Stop loss triggered over time

        
    def simulate_trading(self):
        """
        Currently the core of this class.
        Resets profit and drawdown for evaluation of new individual.
        Simulates trading over the dataset.
        Updates max_profit and drawdown.
        Returned values are used to evaluate the individual (multi-objective).
        """
        self.profit = 0.00 # Reset profit for evaluation of new individual
        self.max_profit = 0.00 # Max profit
        self.drawdown = 0.00 # Reset drawdown for evaluation of new individual
        # Simulate trading over the dataset
        for i in range(len(self.features)):
          
            feature_vector = self.features[i:i+1] # Get the feature vector for the current day

            feature_vector = feature_vector.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            decision = self.model(feature_vector).argmax().item()  # 0=buy, 1=hold, 2=sell

            current_price = self.closing_prices.iloc[i]
            print(f"Decision: {decision}, Current price: {current_price}")
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
        self.max_profit = max(self.max_profit, self.profit)
        self.drawdown = min(self.drawdown, self.profit - self.max_profit)
        print(f"Profit: {self.profit}, Drawdown: {self.drawdown}")
        sleep(1)
        return self.profit, self.drawdown
