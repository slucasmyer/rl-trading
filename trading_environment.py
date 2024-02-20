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

    def __init__(self, features, model, closing_prices, max_gen, pop_size):
        self.features = features  # The dataset
        self.model = model  # The model
        self.closing_prices = closing_prices # Closing prices

        self.initial_balance = 100_000.00  # Initial balance
        self.balance = self.initial_balance  # Balance
        self.max_balance = self.initial_balance  # Max profit
        self.drawdown = 0.00  # Drawdown
        self.shares_owned = 0  # Shares owned
        self.balances = defaultdict(list)  # Balances over time
        self.drawdowns = defaultdict(list)  # Drawdowns over time
        self.stop_loss_triggered = defaultdict(
            list)  # Stop loss triggered over time
        
        self.max_gen = max_gen  # Used to structure dicts
        self.pop_size = pop_size  # Used to structure dicts
        self.current_gen = 1  # Used to identify data
        self.current_ind = 0  # Used to identify data

    def simulate_trading(self):
        """
        Currently the core of this class.
        Resets profit and drawdown for evaluation of new individual.
        Simulates trading over the dataset.
        Updates max_profit and drawdown.
        Returned values are used to evaluate the individual (multi-objective).
        """
        # Update current individual num and current gen num
        self.current_ind += 1
        if self.current_ind > self.pop_size:
            self.current_gen += 1
            self.current_ind = 1
        
        self.balance = self.initial_balance  # Reset profit for evaluation of new individual
        self.max_balance = self.max_balance  # Max profit
        self.drawdown = 0.00  # Reset drawdown for evaluation of new individual
        # Simulate trading over the dataset
        for i in range(len(self.features)):

            # Get the feature vector for the current day
            feature_vector = self.features[i:i+1]

            feature_vector = feature_vector.to(torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"))

            # 0=buy, 1=hold, 2=sell
            decision = self.model(feature_vector).argmax().item()

            current_price = self.closing_prices.iloc[i]
            print(f"Decision: {decision}, Current price: {current_price}")
            #   if decision == 0:  # Buy
            #       self.profit -= self.data['close'][i]
            #   elif decision == 2:  # Sell
            #       self.profit += self.data['close'][i]
            if decision == 0 and self.balance >= current_price:  # Buy
                shares_added = self.balance // current_price
                self.balance -= current_price * shares_added
                self.shares_owned += shares_added

            elif decision == 2 and self.shares_owned > 0:  # Sell
                self.balance += current_price * self.shares_owned
                self.shares_owned = 0

        # Update max_profit and drawdown
        self.max_balance = max(self.max_balance, self.balance)
        self.drawdown = min(self.drawdown, self.balance - self.max_balance)
        print(
            f"Profit: {self.balance - 100_000.00}, Drawdown: {self.drawdown}")
        sleep(1)
        return self.balance - 100_000.00, self.drawdown


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
    print("data_filled", data_filled.head())

    # Convert the DataFrame to a tensor
    features_tensor = torch.tensor(data_filled.values, dtype=torch.float32)

    return features_tensor
