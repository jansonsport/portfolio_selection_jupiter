"""Descriptive price-only return and risk measures for the supplied universe."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def price_panel(prices, start=None, end="2025-12-31"):
    """Return chronological closing prices through the analysis date, without filling gaps."""
    panel = prices.copy()
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.set_index("Date").sort_index()
    if panel.index.has_duplicates:
        raise ValueError("Duplicate trading dates")
    if start is not None:
        panel = panel.loc[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel.loc[panel.index <= pd.Timestamp(end)]
    return panel


def daily_simple_returns(panel):
    """One-session fractional price changes; missing observations stay missing."""
    return panel.pct_change(fill_method=None)


def drawdown_series(panel):
    """Fractional fall from each series' running price peak."""
    return panel.divide(panel.cummax()).subtract(1)


def rolling_annualised_volatility(panel, sessions=63):
    """Rolling standard deviation of daily returns scaled to 252 sessions."""
    return daily_simple_returns(panel).rolling(sessions, min_periods=sessions).std().mul(np.sqrt(TRADING_DAYS))


def risk_summary(prices, start=None, end="2025-12-31", risk_free_rate=0.03):
    """Summarise price-only returns and risk for all tickers against BENCHMARK.

    Arithmetic return and Sharpe use 252 sessions. CAGR uses actual elapsed
    calendar time. The risk-free rate is converted to an effective daily rate.
    """
    panel = price_panel(prices, start, end)
    if "BENCHMARK" not in panel:
        raise ValueError("BENCHMARK is required")
    returns = daily_simple_returns(panel)
    daily_risk_free = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    records = []
    for ticker in panel.columns.drop("BENCHMARK"):
        series = panel[ticker].dropna()
        daily = returns[ticker].dropna()
        paired = returns[[ticker, "BENCHMARK"]].dropna()
        row = {"ticker": ticker, "first_date": pd.NaT, "last_date": pd.NaT,
               "sessions": len(daily), "total_price_return": np.nan,
               "arithmetic_annualised_return": np.nan, "cagr": np.nan,
               "annualised_volatility": np.nan, "max_drawdown": np.nan,
               "sharpe": np.nan, "beta": np.nan, "benchmark_correlation": np.nan,
               "benchmark_relative_return": np.nan}
        if len(series) >= 2:
            row["first_date"], row["last_date"] = series.index[0], series.index[-1]
            elapsed_years = (series.index[-1] - series.index[0]).days / 365.25
            first, last = series.iloc[0], series.iloc[-1]
            if first > 0 and last > 0 and elapsed_years > 0:
                row["total_price_return"] = last / first - 1
                row["cagr"] = (last / first) ** (1 / elapsed_years) - 1
            row["max_drawdown"] = drawdown_series(series).min()
            benchmark = panel["BENCHMARK"].reindex([series.index[0], series.index[-1]])
            if benchmark.notna().all() and (benchmark > 0).all():
                row["benchmark_relative_return"] = (last / first) / (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
        if len(daily) >= 2:
            volatility_daily = daily.std(ddof=1)
            row["arithmetic_annualised_return"] = daily.mean() * TRADING_DAYS
            row["annualised_volatility"] = volatility_daily * np.sqrt(TRADING_DAYS)
            if volatility_daily > 0:
                row["sharpe"] = (daily.mean() - daily_risk_free) / volatility_daily * np.sqrt(TRADING_DAYS)
        if len(paired) >= 2:
            benchmark_variance = paired["BENCHMARK"].var(ddof=1)
            if benchmark_variance > 0:
                row["beta"] = paired[ticker].cov(paired["BENCHMARK"]) / benchmark_variance
                row["benchmark_correlation"] = paired[ticker].corr(paired["BENCHMARK"])
        records.append(row)
    return pd.DataFrame.from_records(records).replace([np.inf, -np.inf], np.nan)
