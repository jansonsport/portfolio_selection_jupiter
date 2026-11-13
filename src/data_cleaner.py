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

    # Correct the known QCK 2025 FCF mismatch using the workbook's cash flows.
    # The source reports 840.0, while 149.51 - 233.51 equals -84.0.
    expected_fcf = (
        cleaned["operating_cash_flow_gbp_m"]
        - cleaned["capital_spending_gbp_m"]
    )
    known_mismatch = (
        cleaned["ticker"].eq("QCK")
        & cleaned["year"].eq(2025)
        & cleaned["free_cash_flow_gbp_m"].eq(840.0)
        & expected_fcf.eq(-84.0)
    )
    cleaned.loc[known_mismatch, "free_cash_flow_gbp_m"] = expected_fcf.loc[
        known_mismatch
    ]

    # Correct the confirmed BSL 2025 profit transcription error.
    bsl_profit_error = (
        cleaned["ticker"].eq("BSL")
        & cleaned["year"].eq(2025)
        & cleaned["profit_gbp_m"].eq(1000.0)
    )
    cleaned.loc[bsl_profit_error, "profit_gbp_m"] = 100.0

    # Preserve sector-specific missing values
    return cleaned

