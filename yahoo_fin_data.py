from yahoo_fin import stock_info


def make_csv():
    """Pulls TQQQ data from Yahoo! Finance with the specified start and end date, drops the 'ticker' column and saves
    the data frame as a CSV."""
    raw_df = stock_info.get_data(ticker="TQQQ", start_date="2011-01-01", end_date="2023-12-31", index_as_date=False)
    raw_df.drop(columns=["ticker"], inplace=True)
    raw_df.to_csv("raw_tqqq_data.csv", index=False)


if __name__ == "__main__":
    make_csv()
