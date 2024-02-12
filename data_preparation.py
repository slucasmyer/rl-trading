from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class DataCollector:
    # Raw values will be still be available after normalizing the data
    data_df: pd.DataFrame = None
    # Data for ML input
    norm_data_df: pd.DataFrame = None
    windows: list[int] = field(default_factory=lambda: [16, 32, 64])
    time_shifts: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])

    def _clean_data(self) -> None:
        """
        Remove set index as the timestamp and drop rows with missing values.
        """
        # Set index as timestamp
        self.data_df = self.data_df.set_index("timestamp")
        self.data_df = self.data_df.dropna()

    def _calculate_stock_measures(self):
        """
        Calculates velocity, acceleration, and average true range.
        """
        # Create stock measurements
        for window in self.windows:
            # Add velocity data to self.data_df
            self._create_velocity_data(window)
            for time_shift in self.time_shifts:
                self._create_velocity_data(window, time_shift)

            # Add acceleration data to self.data_df
            self._create_acceleration_data(window)
            for time_shift in self.time_shifts:
                self._create_acceleration_data(window, time_shift)

            # Add average true range to self.data_df
            self._create_avg_true_range_data(window)
            for time_shift in self.time_shifts:
                self._create_avg_true_range_data(window, time_shift)

    @staticmethod
    def _weighted_moving_avg(close_series: pd.Series, window: int) -> pd.Series:
        """
        Calculate weights to create weighted moving average values.
        """
        # Define weights
        weights = np.arange(1, window + 1)

        # Calculate and return the weighted moving average for the passed close series
        return close_series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    def _hull_moving_avg(self, close_series: pd.Series, window: int) -> pd.Series:
        """
        Creates hull moving average series to be added to self.data_df.
        """
        # Calculate first term
        weighted_half_window = self._weighted_moving_avg(close_series, window // 2)

        # Calculate the second term
        weighted_full_window = self._weighted_moving_avg(close_series, window)

        # Combine the terms and final calculation
        hma_series = 2 * weighted_half_window - weighted_full_window
        hma_series = pd.Series(self._weighted_moving_avg(hma_series, int(np.sqrt(window))), index=close_series.index)

        return hma_series.dropna()

    def _create_velocity_data(self, window: int, time_shift: int = 0) -> None:
        """
        Calculate velocity data and add the series to self.data_df.
        """
        # Branch for window without time shift
        if time_shift == 0:
            close_series = self.data_df['close']
            log_close_series = pd.Series(np.log(close_series), index=close_series.index)
            hma_series = self._hull_moving_avg(log_close_series, window)
            new_series = hma_series.diff()
        # Branch if the data should be time_shifted
        else:
            new_series = pd.Series(self.data_df[f"velocity_{window}w_0ts"].shift(time_shift), index=self.data_df.index)

        # Drop na values from series and add series to self.data_df
        new_series.dropna(inplace=True)
        self.data_df[f"velocity_{window}w_{time_shift}ts"] = new_series

    def _create_acceleration_data(self, window: int, time_shift: int = 0):
        """
        Calculate acceleration data based on the velocity data and add it to self.data_df.
        """
        # Branch for window without time shift
        if time_shift == 0:
            new_series = pd.Series(self.data_df[f"velocity_{window}w_{time_shift}ts"].diff(), index=self.data_df.index)

        # Branch if time shift is present
        else:
            new_series = pd.Series(self.data_df[f"acceleration_{window}w_0ts"].shift(time_shift),
                                   index=self.data_df.index)

        # Drop na values and add series to self.data_df
        new_series.dropna(inplace=True)
        self.data_df[f"acceleration_{window}w_{time_shift}ts"] = new_series

    def _create_avg_true_range_data(self, window: int, time_shift: int = 0):
        """
        Calculate the average true range and add the data to self.data_df
        """
        # Branch if window without time shift
        if time_shift == 0:
            # Collect needed data series
            data_index = self.data_df.index
            high_series = self.data_df['high']
            low_series = self.data_df['low']
            close_prev_series = self.data_df['close'].shift(1)

            # Calculate the true range
            true_range = (
                pd.DataFrame({
                    'h_l': high_series - low_series,
                    'h_c_prev': abs(high_series - close_prev_series),
                    'l_c_prev': abs(low_series - close_prev_series)
                }, index=data_index)
                .max(axis=1)
            )

            # Convert true range to the average true range
            true_range_series = self._hull_moving_avg(true_range, window)

        # Branch if time shift is present
        else:
            true_range_series = self.data_df[f"atr_{window}w_0ts"].shift(time_shift)

        # Add series to self.data_df
        self.data_df[f"atr_{window}w_{time_shift}ts"] = true_range_series

    def _normalize_data(self):
        """
        Normalize all values except for the timestamp column. Min-max scaling to normalize the data between 0 and 1.
        """
        # Extract the timestamp column
        timestamp_column = self.data_df.index

        # Convert the self.data_df to a NumPy array
        data_array = self.data_df.values

        # Normalize all data columns except for the timestamp column
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data_array[:, 1:])

        # Recreate the DataFrame with the normalized data
        self.norm_data_df = pd.DataFrame(normalized_data, index=timestamp_column, columns=self.data_df.columns[1:])

    def _backfill_data(self):
        """
        Backfills cells that do not have a value.
        """
        for column in self.norm_data_df.columns:
            self.norm_data_df[column] = self.data_df[column].interpolate(method='bfill')

    def prepare_and_calculate_data(self) -> pd.DataFrame:
        """
        Adds various stock measurements to determine the velocity, acceleration, and volatility of the asset.
        """
        self._clean_data()
        self._calculate_stock_measures()
        self._normalize_data()
        self._backfill_data()

        return self.norm_data_df


if __name__ == '__main__':
    # Test class and methods
    data = pd.read_csv("training_tqqq.csv")
    # data = pd.read_csv("testing_tqqq.csv")
    data_df = DataCollector(data)
    prepared_data = data_df.prepare_and_calculate_data()

    # Verify results as output
    # print(prepared_data)

    # To save the data
    # prepared_data.to_csv("training_tqqq_prepared.csv")
    # prepared_data.to_csv("testing_tqqq_prepared.csv")
