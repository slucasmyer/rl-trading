from yahoo_fin import stock_info


def get_data(ticker: str, save: bool = False):
    """Pulls TQQQ data from Yahoo! Finance with the specified start and end date, drops the 'ticker' column and can
    save the data frame as a CSV."""
    raw_df = stock_info.get_data(ticker=ticker, start_date="2011-01-01", end_date="2023-12-31", index_as_date=False)
    raw_df.rename(columns={"date": "timestamp"}, inplace=True)
    raw_df.drop(columns=["ticker"], inplace=True)
    if save:
        raw_df.to_csv("raw_tqqq_data.csv", index=False)
    else:
        return raw_df


if __name__ == "__main__":
    get_data("TQQQ")
