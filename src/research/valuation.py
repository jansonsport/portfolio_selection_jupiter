"""Point-in-time equity valuation calculations for the fictional company universe."""

import numpy as np
import pandas as pd


MIN_DENOMINATOR_GBP_M = 1e-6


def valuation_snapshot(annual, prices, as_of="2025-12-31"):
    """Value each company using the latest available annual record and closing price.

    GBP price times millions of shares gives GBP millions of equity value. A
    fiscal-year record enters only after its available_date. Invalid positive
    multiples are NaN; signed earnings and FCF yields preserve losses.
    """
    date = pd.Timestamp(as_of)
    financials = annual.copy()
    financials["available_date"] = pd.to_datetime(financials["available_date"])
    financials = financials.loc[financials["available_date"].le(date)]
    price_history = prices.copy()
    price_history["Date"] = pd.to_datetime(price_history["Date"])
    price_history = price_history.loc[price_history["Date"].le(date)].sort_values("Date")
    if price_history.empty:
        raise ValueError(f"No closing price available by {date.date()}")
    if financials.empty:
        return pd.DataFrame()
    latest = (financials.sort_values(["ticker", "available_date", "year"])
              .drop_duplicates("ticker", keep="last").copy())
    closing = price_history.iloc[-1]
    latest["price_date"] = closing["Date"]
    latest["share_price_gbp"] = latest["ticker"].map(closing.drop("Date"))
    latest["market_value_gbp_m"] = latest["share_price_gbp"] * latest["shares_m"]
    market = latest["market_value_gbp_m"].where(latest["market_value_gbp_m"] > MIN_DENOMINATOR_GBP_M)
    earnings = latest["profit_gbp_m"].where(latest["profit_gbp_m"] > MIN_DENOMINATOR_GBP_M)
    sales = latest["sales_gbp_m"].where(latest["sales_gbp_m"] > MIN_DENOMINATOR_GBP_M)
    latest["pe_ratio"] = market / earnings
    latest["earnings_yield"] = latest["profit_gbp_m"] / market
    latest["fcf_yield"] = latest["free_cash_flow_gbp_m"] / market
    latest["dividend_yield"] = latest["dividend_per_share_gbp"] / latest["share_price_gbp"].where(latest["share_price_gbp"] > 0)
    latest["price_to_sales"] = market / sales
    latest["net_debt_gbp_m"] = latest["debt_gbp_m"] - latest["cash_gbp_m"]
    numeric = latest.select_dtypes(include="number").columns
    latest[numeric] = latest[numeric].replace([np.inf, -np.inf], np.nan)
    return latest.sort_values("ticker").reset_index(drop=True)


def annual_valuation_snapshots(annual, prices, start_year=2016, end_year=2025):
    """Return each year-end snapshot using only financial data then available."""
    snapshots = []
    for year in range(start_year, end_year + 1):
        snapshot = valuation_snapshot(annual, prices, f"{year}-12-31")
        if not snapshot.empty:
            snapshot["valuation_year"] = year
            snapshots.append(snapshot)
    return pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()
