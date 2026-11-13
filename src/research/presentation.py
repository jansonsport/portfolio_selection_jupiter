"""Presentation helpers for cross-company research tables."""

import math
import pandas as pd


METRIC_DIRECTIONS = {
    "revenue_cagr_5y": "higher",
    "profit_margin": "higher",
    "fcf_margin": "higher",
    "net_debt_to_fcf": "lower",
    "annualised_volatility": "lower",
    "max_drawdown": "higher",  # Drawdown is negative; closer to zero is less severe.
    "cagr": "higher",
    "pe_ratio": "neutral",
    "price_to_sales": "neutral",
}


def extreme_tickers(frame: pd.DataFrame, metric: str, direction: str, fraction: float = 0.10):
    """Return deterministic high/low sets among valid values, breaking ties by ticker."""
    if direction not in {"higher", "lower", "neutral"}:
        raise ValueError("direction must be higher, lower or neutral")
    valid = frame.loc[pd.to_numeric(frame[metric], errors="coerce").notna(), ["ticker", metric]].copy()
    valid = valid.loc[pd.to_numeric(valid[metric], errors="coerce").map(math.isfinite)]
    if len(valid) < 10:
        return set(), set()
    count = max(1, math.ceil(len(valid) * fraction))
    ascending = valid.sort_values([metric, "ticker"], ascending=[True, True])["ticker"].tolist()
    descending = valid.sort_values([metric, "ticker"], ascending=[False, True])["ticker"].tolist()
    if direction == "higher":
        return set(descending[:count]), set(ascending[:count])
    if direction == "lower":
        return set(ascending[:count]), set(descending[:count])
    return set(descending[:count]), set(ascending[:count])


def style_extremes(frame: pd.DataFrame, directions: dict[str, str]):
    """Return a readable HTML table with deterministic, direction-aware extremes.

    This avoids pandas Styler's optional Jinja2 dependency. Neutral valuation
    extremes use blue/purple rather than favourable/adverse colours.
    """
    from html import escape
    from IPython.display import HTML

    highlighted = {metric: extreme_tickers(frame, metric, direction)
                   for metric, direction in directions.items() if metric in frame}
    rows = ["<table style='border-collapse:collapse'>",
            "<thead><tr>" + "".join(f"<th style='padding:5px;border-bottom:1px solid #888'>{escape(str(col))}</th>" for col in frame.columns) + "</tr></thead><tbody>"]
    for _, row in frame.iterrows():
        ticker = row["ticker"]
        cells = []
        for col in frame.columns:
            value = row[col]
            label = "—" if pd.isna(value) else (f"{value:,.3f}" if isinstance(value, (float, int)) and not isinstance(value, bool) else str(value))
            colour = ""
            if col in highlighted:
                first, last = highlighted[col]
                if ticker in first:
                    colour = "#c8e6c9" if directions[col] != "neutral" else "#bbdefb"
                elif ticker in last:
                    colour = "#ffcdd2" if directions[col] != "neutral" else "#e1bee7"
            cells.append(f"<td style='padding:5px;border-bottom:1px solid #ddd;background:{colour}'>{escape(label)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return HTML("".join(rows))
