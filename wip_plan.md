# WIP Plan

## Data collection
1. Collect historical data for TQQQ
  1. Choose API
  2. Ensure TQQQ for time-period is available
  3. Decide which value(s) to focus on (open, high, low, close) **Most likely close
  4. Determine the structure for the data so data handling is clear
2. Clean the data for missing data or outliers
  1. Remove or account for missing data points
  2. Check for duplicate values
  3. ID outliers and decide to drop or transform the point 
3. Data visualization
  1. Evaluate raw data for any trends or other useful facts
  2. Determine size of regimes, etc.
4. Divide the data into training (2011-2021) and testing (2022-2023)

## Feature Engineering
1. Calculate RSI/SMA/ADX (other measures/indicators?) values to trigger the buy/sell decisions
2. Explore other features/indices
3. Normalize and/or scale features as needed
  1. Z-score normalization
  2. Min-Max scaling

## Architecture
1. **Neural network(s)**
  1. Design neural network to predict buy/sell based on indices
    1. Input features from above
    2. Design output layer with 3 neurons, 1 for each possible decision
    3. Activation function???
      1. Softmax function might be an option
        1. Example code for `tensorflow` below
    4. Determine loss function - cross-entropy assuming the signal should be a classification problem
  2. Train the NN with the training data
    1. Experiment with number of hidden layers and number of neurons
    2. Choose activation function for the hidden layers (tanh/ReLU?)
  3. Determine the threshold for decision
    1. High-value: buy, middle-range: hold, low-value: sell
    2. Example: value >= 0.65 -> buy
  4. Use LSTMs or RNNs? (Thought we're not using these anymore?)
    1. RNN is simpler than LSTMs while LSTMs work great for time-sequence data (like stocks)
    2. Would require transforming the input data into windows of time sequences
    3. Reshape input into 3D array
      1. `(num_records, num_windows, num_features)`
    4. See below for alternative code using LSTM 
2. **Genetic algorithm**
  1. Optimize for low drawdown and high profitability
  2. Define chromosomes
    1. Based on features, risk management plan
    2. Convert NN prediction into decisions based on the thresholds
  3. Define a function that creates a fitness score
  4. Create a starting population
  5. Run genetic algorithm
  6. Find the Pareto frontier
  7. Allow for selecting which point along the frontier is desired by the user based on risk

## Evaluate
1. Decide on performance metrics
2. Test model on training data
  1. Check for overfitting, other issues
  2. Study differences in baseline algorithm trading strategy and model produced data
3. Test model on test data
  1. Accuracy
  2. Profitability
  3. Max draw down
4. Iteratively work on model architecture

## Risk Management
1. Define as a stop-loss value (number or percent) 
  1. Or by the chosen optimization model from the genetic algorithm?
  2. What is the max acceptable loss allowed before selling?
  3. Constantly monitor checking for stop-loss trigger?
2. Define a take-profit threshold to ensure profit?

## Deploy trading agent?
1. Connect to trading API
2. Monitor performance
3. Retrain with more recent data (months or years later)?

## Documentation
1. Overview
2. Data collection process
3. Preprocessing
4. Model architecture
5. Training process
6. Hyperparameter tuning
7. Integration with genetic algorithm
8. Risk Management
9. Results and Performance
10. Projected next steps
11. Code
  1. Dependencies and environment
12. Challenges
13. Acknowledgements
14. References
15. Legal disclaimer?


## Example Code
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# num_input_features would be the vector(array) of feature (close, SMA, RSI, ADX, etc.) and date data

# Model
model = Sequential()  # initialize an empty sequential model
model.add(Dense(units=hidden_units, input_dim=num_input_features, activation='relu))  # hidden layer using ReLU for the activation function
model.add(Dense(units=3))  # Output layer has 3 neurons (buy/sell/hold)
model.add(Activation('softmax'))  # Used for multi-class classification


# Alternative using LSTM instead of ReLU activation function
from tensorflow.keras.layers import LSTM

# Variables needed for windows and the number of features
num_windows = ?
num_features = ?

# Transform input data
input_reshaped = input.reshape((input.shape[0], num_windows, num_features))

model = Sequential()  # initialize an empty sequential model
model.add(LSTM(units=50, input_shape(num_windows, num_features), return_sequences=True))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=3, activation='softmax'))
```
