import pandas as pd
import numpy as np


def validate_daily_prices(df):
    checks = {}

    # 1. Duplicate dates
    checks["duplicate_dates"] = df["Date"].duplicated().sum()

    # 2. Invalid dates
    checks["invalid_dates"] = df["Date"].isna().sum()

    # 3. Missing price observations
    checks["missing_prices"] = (
        df.drop(columns="Date").isna().sum().sum()
    )

    # 4. Non-positive prices
    checks["invalid_prices"] = (
        (df.drop(columns="Date") <= 0).sum().sum()
    )

    # 5. Chronological ordering
    checks["dates_sorted"] = df["Date"].is_monotonic_increasing

    return checks