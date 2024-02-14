import numpy as np
import pandas as pd

class TradeEnvironment:
    def __init__(self, data_df):
        self._current_balance = 100000
        self._num_stocks = 0  # int only
        self._data_df = data_df  # This is either cleaned training data or cleaned testing data
        self._current_profit = 0  # Can be negative
        self._is_invested = False

    def get_current_balance(self):
        return self._current_balance

    def get_num_stocks(self):
        return self._num_stocks

    def get_data_df(self):
        return self._data_df

    def get_current_profit(self):
        return self._current_profit

    def get_is_invested(self):
        return self._is_invested

    def set_current_balance(self, new_balance):
        self._current_balance = new_balance

    def set_num_stocks(self, new_stocks):
        self._num_stocks = new_stocks

    def set_current_profit(self, new_profit):
        self._current_profit = new_profit

    def set_is_invested(self, new_bool):
        self._is_invested = new_bool

    def buy_stocks(self):
        """Updates current balance and num stocks by buying stocks"""
        pass

    def sell_stocks(self):
        """Updates current balance and num stocks by selling stocks"""
        pass

    def portfolio_output(self):
        """Will show current balance, profit, number of stocks, investment status, and investment status
        (How often will vary, probably every day?)"""
        pass

    def evaluate_nn_decision(self, decision: int) -> int:
        """Used to check decision returned by neural network. Depending on stop loss factors and other factors that
        can override the neural network decision."""
        # return final_decision

    def update_portfolio(self, decision: int):
        """Updates portfolio based on decision (0, 1, or 2) 0 = Buy, 1 = Hold, 2 = Sell"""
        if decision == 0:
            self.buy_stocks()

        elif decision == 2:
            self.sell_stocks()

        else:
            # Any additional attributes that need updating or actions to be taken?
            pass


if __name__ == "__main__":
    # env = TradeEnvironment()  # Cleaned data df to be loaded here....
    # Call NN here and return value is 0, 1, or 2?  # With cleaned data df, input_size = 15, output_size = 3
    pass


