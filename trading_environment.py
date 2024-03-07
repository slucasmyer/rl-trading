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
        self.profit: float = 0.00 # Profit percentage
        self.num_trades = 0  # Number of trades
        self.balances = defaultdict(list)  # Balances over time
        self.drawdowns = defaultdict(list)  # Drawdowns over time
        self.decisions = defaultdict(list) # Decisions over time
        self.stop_loss_triggered = defaultdict(list)  # Stop loss triggered over time
        
        self.max_gen = max_gen  # Used to structure dicts
        self.pop_size = pop_size  # Used to structure dicts
        self.current_gen = 1  # Used to identify data
        self.current_ind = 0  # Used to identify data

    def simulate_trading(self, chromosome: int):
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
        self.shares_owned = 0  # Reset shares owned for evaluation of new individual
        self.profit = 0.00  # Reset profit for evaluation of new individual
        self.num_trades = 0  # Reset number of trades for evaluation of new individual
        local_decisions = []
        # Simulate trading over the dataset
        for i in range(len(self.features)):
            
            feature_vector = self.features[i:i+1] # Get the feature vector for the current day

            feature_vector = feature_vector.to(torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"))

            decision = self.model(feature_vector).argmax().item()  # 0=buy, 1=hold, 2=sell
            local_decisions.append(decision)
            current_price = self.closing_prices.iloc[i]
            if decision == 0 and self.balance >= current_price:  # Buy
                shares_bought = self.balance // current_price
                self.shares_owned += shares_bought
                self.balance -= current_price * shares_bought
                self.num_trades += 1
                
            elif decision == 2 and self.shares_owned > 0:  # Sell
                self.balance += current_price * self.shares_owned
                self.shares_owned = 0
                self.num_trades += 1

            self.max_balance = max(self.max_balance, self.balance)
            current_drawdown = self.balance + (self.shares_owned * current_price) - self.max_balance
            self.drawdown = min(self.drawdown, current_drawdown)

        self.balances[chromosome].append(self.balance)
        self.drawdowns[chromosome].append(self.drawdown)
        self.decisions[chromosome].append(decision)
        self.stop_loss_triggered[chromosome].append(0)
        
        if self.shares_owned > 0:
            self.balance += self.shares_owned * self.closing_prices.iloc[-1]
            self.shares_owned = 0
        self.profit = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        self.drawdown = (self.drawdown / self.initial_balance) * 100
        # print(f"simulate_trading Gen {self.current_gen}, Pop {self.current_ind}, Profit: {self.profit}, Drawdown: {self.drawdown}, Num Trades: {self.num_trades}")
        # sleep(0.5)
        return self.profit, self.drawdown, float(self.num_trades)


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
    data_filled = data.bfill().ffill().fillna(0)

    # Convert the DataFrame to a tensor
    features_tensor = torch.tensor(data_filled.values, dtype=torch.float32)

    return features_tensor
