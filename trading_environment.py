import pandas as pd
import torch


class TradingEnvironment:
    """
    A class to simulate trading in a stock market.
    The environment is initialized with a dataset and a model.
    The model is used to make trading decisions based on the dataset.
    Profit and drawdown are calculated based on the trading decisions.
    """

    def __init__(self, features, model, closing_prices):
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
        
    def reset(self):
        """Resets the environment."""
        self.balance = self.initial_balance
        self.max_balance = self.balance
        self.drawdown = 0.00
        self.shares_owned = 0
        self.profit = 0.00
        self.num_trades = 0

    def set_model(self, new_model):
        """Sets the model."""
        self.model = new_model

    def set_features(self, new_features):
        """Sets the features."""
        self.features = new_features

    def set_closing_prices(self, new_closing_prices):
        """Sets the closing prices."""
        self.closing_prices = new_closing_prices

    def simulate_trading(self):
        """
        Currently the core of this class.
        Resets profit and drawdown for evaluation of new individual.
        Simulates trading over the dataset.
        Updates max_profit and drawdown.
        Returned values are used to evaluate the individual (multi-objective).
        """
        
        self.reset()
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

            current_portfolio_value = self.balance + (self.shares_owned * current_price)
            self.max_balance = max(self.max_balance, current_portfolio_value)
            current_drawdown = self.max_balance - current_portfolio_value if current_portfolio_value < self.max_balance else 0.00
            drawdown_pct = (current_drawdown / self.max_balance) * 100
            self.drawdown = max(self.drawdown, drawdown_pct)

        
        if self.shares_owned > 0:
            self.balance += self.shares_owned * self.closing_prices.iloc[-1]
            self.shares_owned = 0
            self.num_trades += 1
        
        raw_profit = self.balance - self.initial_balance
        scaled_profit = raw_profit / self.initial_balance
        profit_pct = scaled_profit * 100
        self.profit = profit_pct

        return self.profit, self.drawdown, float(self.num_trades)
