"""Read and write the project's cleaned datasets."""

from pathlib import Path

import pandas as pd


ANNUAL_FILENAME = "annual_figures.csv"
PRICES_FILENAME = "daily_prices.csv"


def save_cleaned_data(annual, prices, directory):
    """Save the approved cleaned tables as CSV files and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    annual_path = directory / ANNUAL_FILENAME
    prices_path = directory / PRICES_FILENAME
    annual.to_csv(annual_path, index=False, date_format="%Y-%m-%d")
    prices.to_csv(prices_path, index=False, date_format="%Y-%m-%d")
    return annual_path, prices_path


def load_cleaned_data(directory):
    """Load saved tables with their date columns parsed as dates."""
    directory = Path(directory)
    annual = pd.read_csv(directory / ANNUAL_FILENAME, parse_dates=["available_date"])
    prices = pd.read_csv(directory / PRICES_FILENAME, parse_dates=["Date"])
    return annual, prices
