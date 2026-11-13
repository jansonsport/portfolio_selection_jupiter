"""Point-in-time annual fundamental calculations, with undefined ratios left missing."""

import numpy as np
import pandas as pd


MIN_DENOMINATOR = 0.01  # GBP millions, or GBP per share for EPS growth.
SPECIAL_FINANCIALS = {"CDB", "BSL", "CRM"}


def _positive_denominator(values, threshold=MIN_DENOMINATOR):
    """Return positive denominators above the documented materiality floor."""
    return values.where(values > threshold)


def calculate_fundamentals(annual):
    """Return annual amounts and interpretable ratios for all available records."""
    data = annual.copy().sort_values(["ticker", "year"]).reset_index(drop=True)
    data["available_date"] = pd.to_datetime(data["available_date"])
    group = data.groupby("ticker", sort=False)
    sales = _positive_denominator(data["sales_gbp_m"])
    shares = _positive_denominator(data["shares_m"])
    profit = data["profit_gbp_m"]
    fcf = data["free_cash_flow_gbp_m"]

    data["revenue_growth"] = data["sales_gbp_m"] / _positive_denominator(group["sales_gbp_m"].shift()) - 1
    prior_profit = group["profit_gbp_m"].shift()
    data["profit_growth"] = profit.where(profit > MIN_DENOMINATOR) / _positive_denominator(prior_profit) - 1
    data["eps_gbp"] = profit / shares
    prior_eps = data.groupby("ticker")["eps_gbp"].shift()
    data["eps_growth"] = data["eps_gbp"].where(data["eps_gbp"] > MIN_DENOMINATOR) / _positive_denominator(prior_eps) - 1
    prior_shares = group["shares_m"].shift()
    data["share_count_growth"] = data["shares_m"] / _positive_denominator(prior_shares) - 1
    first_shares = group["shares_m"].transform("first")
    data["share_count_change_since_start"] = data["shares_m"] / _positive_denominator(first_shares) - 1

    data["profit_margin"] = profit / sales
    data["average_profit_margin_5y"] = data.groupby("ticker")["profit_margin"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    data["profit_margin_std_5y"] = data.groupby("ticker")["profit_margin"].transform(lambda s: s.rolling(5, min_periods=3).std())
    data["fcf_margin"] = fcf / sales
    data["ocf_to_profit"] = data["operating_cash_flow_gbp_m"] / _positive_denominator(profit)
    data["capex_to_revenue"] = data["capital_spending_gbp_m"] / sales
    data["net_debt_gbp_m"] = data["debt_gbp_m"] - data["cash_gbp_m"]
    industrial = ~data["ticker"].isin(SPECIAL_FINANCIALS)
    data["net_debt_to_fcf"] = (data["net_debt_gbp_m"] / _positive_denominator(fcf)).where(industrial)
    data["net_debt_to_profit"] = (data["net_debt_gbp_m"] / _positive_denominator(profit)).where(industrial)
    data["dividend_payout"] = (data["dividend_per_share_gbp"] / _positive_denominator(data["eps_gbp"])).where(profit > MIN_DENOMINATOR)

    for years in (3, 5, 9):
        previous = data.groupby("ticker")["sales_gbp_m"].shift(years)
        previous_year = data.groupby("ticker")["year"].shift(years)
        valid = data["year"].sub(previous_year).eq(years)
        data[f"revenue_cagr_{years}y"] = ((data["sales_gbp_m"] / _positive_denominator(previous)) ** (1 / years) - 1).where(valid)

    for column in ("loans_overdue", "capital_cushion", "claims_ratio", "capital_coverage"):
        data[f"{column}_change_pp"] = data.groupby("ticker")[column].diff() * 100
    return data.replace([np.inf, -np.inf], np.nan)


def latest_fundamentals(annual, as_of="2025-12-31"):
    """Return the latest known record per company on the analysis date."""
    known = annual.loc[pd.to_datetime(annual["available_date"]) <= pd.Timestamp(as_of)]
    calculated = calculate_fundamentals(known)
    return calculated.groupby("ticker", sort=False).tail(1).sort_values("ticker").reset_index(drop=True)


def compare_periods(annual, start_year=2020, end_year=2025):
    """Compare annual amounts and margins without converting sign changes to growth rates."""
    calculated = calculate_fundamentals(annual)
    columns = ["sales_gbp_m", "profit_gbp_m", "profit_margin", "operating_cash_flow_gbp_m", "free_cash_flow_gbp_m", "fcf_margin", "net_debt_gbp_m", "shares_m"]
    selected = calculated.loc[calculated["year"].isin([start_year, end_year]), ["ticker", "year", *columns]]
    wide = selected.set_index(["ticker", "year"])[columns].unstack("year")
    wide.columns = [f"{metric}_{year}" for metric, year in wide.columns]
    return wide.reset_index()
