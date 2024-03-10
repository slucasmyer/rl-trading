# Stock Market AI

This is an artificial intellegence developed via PyTorch that simulates stock trading using the TQQQ(UltraPro QQQ) security. The agent was taught using reinforcement learning with training and testing data extracted from [yahoo_fin](https://theautomatic.net/yahoo_fin-documentation/).

The agent's goal is to maximize profit and minimize drawdown by simulating stock trade of the TQQQ with historical stock data from 2022-01-01 to 2023-12-31. Training data consists of using historical stock data from 2011-01-01 to 2021-12-31.

Data visualizations are shown at the end that displays profits and drawdowns as well as stop loss triggers for the top ten best solution candidates based on what the agent has chosen. These solution candidates have been optimized via the NSGA-2 Algorithm(Non-Sorted Genetic Algorithm) via the PyMoo library.

# Developers

- Sullivan Myer
- Steven Crowther
- Andrew Perez
- Alexander Licato

# Key Libraries, APIs, and Technologies Used

- [PyTorch](https://pytorch.org/)
- [PyMoo](https://pymoo.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [MatPlotLib](https://matplotlib.org/)
- [Google Colab](https://colab.google/)
- [yahoo_fin](https://theautomatic.net/yahoo_fin-documentation/)

# Workflow

[General Flowchart](https://github.com/slucasmyer/rl-trading/blob/3b50608b826a152888bd7b09e89c3b1c90897fff/Current%20Workflow.png)

# main.py

The main script of the artifical intellegence. A CSV that contains raw data from the Alpha Vantage API is loaded into data_preperation.py so that the data gets converted into inputs that can be used by trading_environment.py and other modules.

The PolicyNetwork object then gets instantiated. There are checks in place to see if the machine this script is running on contains CUDA-compatible GPUs. If not, then the CPU is used. Other objects are instantiated as well including TradingEnvironment and TradingProblem. Thread pools are created as well as the runner, which is used for ElementWiseProblem parallelization used by the TradingProblem object. The algorithm object is then intialized, which utlizes the NSGA-2 algorithm used by the minimized optimization object.

The training and or testing begins with a call to minimize() from the PyMoo module. This function utilizes all of the instiatited objects mentioned above for the learning to begin. At the end, the results are plotted.

# yahoo_fin_data.py

The script used to extract historical price data for the TQQQ. A call is made to the Yahoo! Finance API, which contains historical daily high price, low price, opening price, closing price, and volume values of the TQQQ throughout its entire history.

The API call returns a Pandas dataframe containing this data. The "ticker" column is dropped, as that column is not necessary and certain column names are
changed for compatiblility with data_preperation.py.

CSVs are saved in the same directory as yahoo_fin_data.py if the user decides to make the optional parameter "bool" equal to true.

# data_preparation.py

The script used to convert raw price data from CSVs into a DataFrame object to be used by the agent. The dataframe will contain the following columns\*:

\*All values for each column are normalized except for Timestamp

- Timestamp (YYYY-MM-DD)
- Open
- High
- Low
- Close
- Volume
- Velocity
  - 16 day, 32 day, 64 day windows
  - 2 day, 4 day, 6 day, 8 day, 10 day time shifts
- Acceleration
  - 16 day, 32 day, 64 day windows
  - 2 day, 4 day, 6 day, 8 day, 10 day time shifts
- Average True Range (Volatility)
  - 16 day, 32 day, 64 day windows
  - 2 day, 4 day, 6 day, 8 day, 10 day time shifts

# trading_environment.py

The script used to create the trading environment that the agent uses to make its trading decisions. The environment includes the simulated portfolio that includes the current balance of the portfolio and number of stocks owned. The agent starts each session with a balance of $100,000. The environment iterates through the preprocessed data dataframe and feeds the agent a PyTorch tensor containing the stock data for that particular day. The agent returns one of three action decisions: buy stock(0), hold/do nothing(1), and sell stock(2).

These decisions must be evaluated to see if stop-loss triggers will occur. If no stop-loss trigger has occured, then the decision will go through and update the portfolio balances as necessary. If a stop-loss trigger has occured, the initial decision is overruled and a sell action will be initiated, updating the balance as necessary.

# policy_network.py

The script that creates the simple feed-forward neural network for the artificial intelligence. There are 6 linear layers, which also include activation layers utilizing the ReLU activation function. The first layer consists of 10 nodes, which are the inputs to the neural network. The second layer consists of 5 nodes, the third layer with 10 nodes, the fourth layer with 4 nodess, the fifth layer with 10 nodes, and the sixth and final layer with 3 nodes. The final 3 nodes represent the output for the neural network, which is either a 0 (buy), 1 (hold), or 2(sell) decision.

# trading_problem.py

The script that creates the TradingProblem class for the multi-objective optimization for the agent. Utilizes the PolicyNetwork and TradingEnvironment objects. The script decodes the genes of an individual chromosome to feed into the neural network.

# plotter.py

The script that pre-processes results from the agent's training/testing and plots them into graphs. There are two graphs, a profit vs. drawdown 2-Dimensional graph and a profit vs. drawdown. vs trade count 3-Dimensional graph. Each point represents an individual solution in each generation. A color gradient indicates which generation the solution came from, with yellow starting at generation 1 to dark purple being at generation 50.

![Example Profit vs. Drawdown 2D](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")
