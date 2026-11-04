import pandas as pd
import numpy as np


def clean_daily_prices(df):
    cleaned = df.copy()

    # Convert dates
    cleaned["Date"] = pd.to_datetime(
        cleaned["Date"], errors="raise"
    )

    # Sort chronologically
    cleaned = cleaned.sort_values("Date").reset_index(drop=True)

    # Check unique trading dates
    if cleaned["Date"].duplicated().any():
        raise ValueError("Duplicate trading dates detected")

    # Check for invalid prices
    price_columns = cleaned.columns.drop("Date")

    if (cleaned[price_columns] <= 0).any().any():
        raise ValueError("Zero or negative prices detected")

    # Flag the known suspicious MGS observation
    anomaly_date = pd.Timestamp("2024-06-17")
    mask = cleaned["Date"].eq(anomaly_date)

    cleaned.loc[mask, "MGS"] /= 100

    return cleaned


def clean_annual_figures(df):
    cleaned = df.copy()

    # Convert availability dates
    cleaned["available_date"] = pd.to_datetime(
        cleaned["available_date"], errors="raise"
    )

    # Check unique company-year combinations
    if cleaned.duplicated(["ticker", "year"]).any():
        raise ValueError("Duplicate company-year records detected")

    # Sort financial observations
    cleaned = cleaned.sort_values(
        ["ticker", "year"]
    ).reset_index(drop=True)

    # Preserve sector-specific missing values
    return cleaned

