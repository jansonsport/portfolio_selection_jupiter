import pandas as pd
import numpy as np

from src.outlier_checks import neighboring_year_outliers


def validate_daily_prices(df):
    checks = {}

    price_data = df.drop(columns="Date")

    # 1. Duplicate dates
    checks["duplicate_dates"] = df["Date"].duplicated().sum()

    # 2. Invalid dates
    checks["invalid_dates"] = df["Date"].isna().sum()

    # 3. Missing price observations
    checks["missing_prices"] = price_data.isna().sum().sum()

    # 4. Non-positive prices
    checks["invalid_prices"] = (price_data <= 0).sum().sum()

    # 5. Chronological ordering
    checks["dates_sorted"] = df["Date"].is_monotonic_increasing

    # 6. Infinite price values
    checks["infinite_prices"] = np.isinf(
        price_data.to_numpy(dtype=float)
    ).sum()

    return checks


def validate_annual_figures(df):
    checks = {}

    # 1. Duplicate company-year records
    checks["duplicate_company_years"] = (
        df.duplicated(["ticker", "year"]).sum()
    )

    # 2. Invalid availability dates
    checks["invalid_dates"] = df["available_date"].isna().sum()

    # 3. Missing company identifiers
    checks["missing_tickers"] = df["ticker"].isna().sum()

    # 4. Invalid shares outstanding
    checks["invalid_shares"] = (df["shares_m"] <= 0).sum()

    # 5. Free cash flow consistency
    # FCF = Operating Cash Flow - Capital Spending
    fcf_difference = (
        df["operating_cash_flow_gbp_m"]
        - df["capital_spending_gbp_m"]
        - df["free_cash_flow_gbp_m"]
    )

    # Allow small differences due to rounding
    checks["fcf_mismatches"] = (
        fcf_difference.abs() > 0.02
    ).sum()

    # Known source transcription error: BSL 2025 profit is 100, not 1000.
    checks["bsl_2025_profit_errors"] = (
        df["ticker"].eq("BSL")
        & df["year"].eq(2025)
        & df["profit_gbp_m"].eq(1000.0)
    ).sum()

    # Exploratory review count: a flag is not proof of a data error.
    checks["neighboring_year_review_flags"] = len(neighboring_year_outliers(df))

    return checks
