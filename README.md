# Stock Market AI

This is an artificial intellegence developed via PyTorch that simulates stock trading using the TQQQ(UltraPro QQQ) security. The agent was taught using reinforcement learning with training and testing data extracted from [Alpha Vantage's API](https://www.alphavantage.co/documentation/). The agent's goal is to maximize profit and minimize drawdown by simulating stock trade of the TQQQ with historical stock data from 2022-01-01 to 2023-12-31. Training data consists of using historical stock data from 2011-01-01 to 2021-12-31. Data visualizations are shown at the end that displays profits and drawdowns as well as stop loss triggers for the top ten best solution candidates based on what the agent has chosen. These solution candidates have been optimized via the NSGA-2 Algorithm(Non-Sorted Genetic Algorithm) via the PyMoo library.

# Key Libraries, APIs, and Technologies Used

- [PyTorch](https://pytorch.org/)
- [PyMoo](https://pymoo.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [MatPlotLib](https://matplotlib.org/)
- [Google Colab](https://colab.google/)
- [Alpha Vantage API](https://www.alphavantage.co/documentation/)

# Workflow
