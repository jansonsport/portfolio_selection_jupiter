"""Exploratory outlier screens for the historical finance workbook.

These functions flag observations for review; they do not decide that a value
is erroneous and never alter the supplied DataFrames.
"""

import numpy as np
import pandas as pd


def robust_return_outliers(prices, threshold=6.0):
    """Flag ticker returns far from that ticker's median using scaled MAD."""
    data = prices.sort_values("Date").set_index("Date").drop(columns="BENCHMARK", errors="ignore")
    returns = data.pct_change(fill_method=None)
    records = []
    for ticker in returns:
        values = returns[ticker].dropna()
        median = values.median()
        mad = (values - median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            continue
        score = 0.67448975 * (values - median) / mad
        for date in score.index[score.abs() >= threshold]:
            records.append({"ticker": ticker, "date": date, "metric": "daily_return",
                            "value": returns.at[date, ticker], "score": score.at[date],
                            "reason": f"absolute robust score >= {threshold:g}"})
    return pd.DataFrame(records, columns=["ticker", "date", "metric", "value", "score", "reason"])


def annual_outliers(annual, threshold=6.0):
    """Flag extreme common financial values and within-ticker annual changes."""
    metrics = ["sales_gbp_m", "profit_gbp_m", "operating_cash_flow_gbp_m",
               "capital_spending_gbp_m", "free_cash_flow_gbp_m", "debt_gbp_m",
               "cash_gbp_m", "shares_m"]
    records = []
    for metric in metrics:
        for ticker, group in annual.groupby("ticker"):
            values = group[metric].dropna()
            if len(values) < 4:
                continue
            med, mad = values.median(), (values - values.median()).abs().median()
            if mad and np.isfinite(mad):
                scores = 0.67448975 * (values - med) / mad
                for idx in scores.index[scores.abs() >= threshold]:
                    row = annual.loc[idx]
                    records.append({"ticker": ticker, "date": row["year"], "metric": metric,
                                    "value": row[metric], "score": scores.loc[idx],
                                    "reason": f"within-ticker robust level score >= {threshold:g}"})
    ordered = annual.sort_values(["ticker", "year"])
    for metric in metrics:
        change = ordered.groupby("ticker")[metric].pct_change(fill_method=None)
        for ticker, vals in change.groupby(ordered["ticker"]):
            vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) < 4:
                continue
            med, mad = vals.median(), (vals - vals.median()).abs().median()
            if mad and np.isfinite(mad):
                scores = 0.67448975 * (vals - med) / mad
                for idx in scores.index[scores.abs() >= threshold]:
                    row = ordered.loc[idx]
                    records.append({"ticker": ticker, "date": row["year"], "metric": f"{metric}_yoy_change",
                                    "value": change.loc[idx], "score": scores.loc[idx],
                                    "reason": f"within-ticker robust annual-change score >= {threshold:g}"})
    return pd.DataFrame(records, columns=["ticker", "date", "metric", "value", "score", "reason"])


def neighboring_year_outliers(annual, threshold=3.0):
    """Flag numeric annual values far from adjacent fiscal years for each ticker.

    Compare only consecutive years with available values. The reference is the
    mean of one or two neighbors, and the scale is the larger of its absolute
    value and the ticker's median absolute consecutive-year change.
    """
    metrics = [column for column in annual.select_dtypes(include="number")
               if column != "year"]
    records = []
    for ticker, company in annual.sort_values(["ticker", "year"]).groupby("ticker"):
        company = company.reset_index(drop=True)
        years = company["year"].to_numpy()
        for metric in metrics:
            values = pd.to_numeric(company[metric], errors="coerce").to_numpy(dtype=float)
            changes = [abs(values[i] - values[i - 1]) for i in range(1, len(company))
                       if years[i] == years[i - 1] + 1
                       and np.isfinite(values[i]) and np.isfinite(values[i - 1])]
            typical_change = float(np.median(changes)) if changes else 0.0
            for i, value in enumerate(values):
                if not np.isfinite(value):
                    continue
                previous = (values[i - 1] if i > 0 and years[i - 1] == years[i] - 1
                            and np.isfinite(values[i - 1]) else np.nan)
                following = (values[i + 1] if i + 1 < len(values)
                             and years[i + 1] == years[i] + 1
                             and np.isfinite(values[i + 1]) else np.nan)
                neighbors = [v for v in (previous, following) if np.isfinite(v)]
                if not neighbors:
                    continue
                reference = float(np.mean(neighbors))
                difference = abs(value - reference)
                scale = max(abs(reference), typical_change)
                if difference > 0 and difference >= threshold * scale:
                    records.append({"ticker": ticker, "date": int(years[i]),
                                    "metric": metric, "value": value, "score": np.nan,
                                    "previous_value": previous, "next_value": following,
                                    "reference_value": reference,
                                    "typical_change": typical_change,
                                    "reason": f"distance from adjacent year(s) >= {threshold:g} times local scale"})
    return pd.DataFrame(records, columns=["ticker", "date", "metric", "value", "score",
                                          "previous_value", "next_value", "reference_value",
                                          "typical_change", "reason"])


def known_annual_errors(annual):
    """Flag confirmed source values before their targeted cleaning corrections."""
    mask = (annual["ticker"].eq("BSL") & annual["year"].eq(2025)
            & annual["profit_gbp_m"].eq(1000.0))
    records = [{"ticker": "BSL", "date": 2025, "metric": "profit_gbp_m",
                "value": row["profit_gbp_m"], "score": np.nan,
                "reason": "confirmed source error; expected 100.0 GBP million"}
               for _, row in annual.loc[mask].iterrows()]
    return pd.DataFrame(records, columns=["ticker", "date", "metric", "value", "score", "reason"])


def price_discontinuities(prices, stale_days=5):
    """Flag large daily gaps and runs of unchanged closes; returns review rows."""
    data = prices.sort_values("Date").set_index("Date").drop(columns="BENCHMARK", errors="ignore")
    returns = data.pct_change(fill_method=None)
    records = []
    for ticker in data:
        vals = returns[ticker].dropna()
        med = vals.median()
        mad = (vals - med).abs().median()
        if mad and np.isfinite(mad):
            scores = 0.67448975 * (vals - med) / mad
            for date in scores.index[scores.abs() >= 6]:
                records.append({"ticker": ticker, "date": date, "metric": "price_gap_return",
                                "value": returns.at[date, ticker], "score": scores.at[date],
                                "reason": "robust return discontinuity (inspect adjacent closes)"})
        same = data[ticker].eq(data[ticker].shift())
        run = same.groupby((~same).cumsum()).transform("sum")
        for date in data.index[run >= stale_days]:
            records.append({"ticker": ticker, "date": date, "metric": "unchanged_close_run",
                            "value": int(run.loc[date]), "score": np.nan,
                            "reason": f"unchanged close for at least {stale_days} consecutive observations"})
    return pd.DataFrame(records, columns=["ticker", "date", "metric", "value", "score", "reason"])


def annual_ratio_flags(annual):
    """Review unusual common-field ratios without imposing sector-wide rules."""
    data = annual.copy()
    sales = data["sales_gbp_m"].where(data["sales_gbp_m"].ne(0))
    ratios = {
        "profit_margin": data["profit_gbp_m"] / sales,
        "operating_cash_flow_to_sales": data["operating_cash_flow_gbp_m"] / sales,
        "free_cash_flow_to_sales": data["free_cash_flow_gbp_m"] / sales,
        "debt_to_cash": data["debt_gbp_m"] / data["cash_gbp_m"].where(data["cash_gbp_m"].ne(0)),
    }
    records = []
    for metric, values in ratios.items():
        for idx, value in values.items():
            if pd.notna(value) and (not np.isfinite(value) or abs(value) > 10):
                row = data.loc[idx]
                records.append({"ticker": row["ticker"], "date": row["year"], "metric": metric,
                                "value": value, "score": np.nan,
                                "reason": "ratio outside review band (>|10|) or non-finite"})
    return pd.DataFrame(records, columns=["ticker", "date", "metric", "value", "score", "reason"])
