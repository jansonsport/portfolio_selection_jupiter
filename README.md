# Portfolio Selection

Research project using financial statements and price histories for 20 fictional companies.

## Proposed portfolio

Invest **£100,000** at the end of 2025 and hold through **2028**. Each stock starts at **20% (£20,000)**.

| Ticker | Company | Business exposure |
| --- | --- | --- |
| MGS | Marigold Systems | Essential technology |
| BFC | Bluefin Cyber | Cybersecurity |
| ORP | Oriel Pharma | Routine healthcare |
| CSE | ClearSpan Engineering | Infrastructure spending |
| LNP | Lantern Payments | Payments and consumer spending |

These were the five highest-scoring eligible companies on 31 December 2025. Technology accounts for 40% of the initial portfolio.

## Why these factors?

The aim was to grow the portfolio while limiting large losses. The market briefing pointed to slow growth, weak discretionary demand, and continued refinancing pressure, making stable margins, cash generation, and debt capacity especially relevant. Valuation and price risk helped assess what investors were paying and how sharply shares had fallen in the past.

| Factor | What we measured |
| --- | --- |
| Growth | Three-year revenue growth and latest earnings-per-share growth |
| Profitability | Three-year average profit margin and margin stability |
| Cash generation | Latest and three-year average free cash flow margins |
| Financial resilience | Net debt relative to positive free cash flow |
| Valuation | Earnings and free cash flow yields |
| Low price risk | Trailing 12-month volatility and maximum drawdown |

We converted the measures to percentile ranks and gave each factor one-sixth of the company score. The short history and overlapping factor baskets gave little basis for fine-tuning the weights. Companies needed all six scores to qualify.

## What the backtests showed

We rebuilt each historical selection using only information available at the time. Across five annual periods from April 2020 to April 2025, the combined score produced these results with annual reselection and 0.25% costs per buy or sell.

| Holdings | Annualised price return | Maximum drawdown |
| ---: | ---: | ---: |
| 5 | 35.08% | −5.12% |
| 8 | 31.21% | −7.51% |
| 10 | 28.78% | −7.48% |

The five-stock basket led in this sample. Giving less weight to more volatile stocks barely reduced volatility (7.76% to 7.73%), supporting a simple 20% starting weight for each holding.

To match the proposed holding period, we also tested three overlapping three-year windows. Buy-and-hold returned **24.52%–34.40% annualised** after costs and led annual reselection in one of three windows. The tests exclude dividends, and the overlapping windows provide limited evidence about which holding policy works best.

## Analysis steps

1. [Explore and clean the data](notebooks/data_exploration.ipynb). Check dates, duplicates, missing values, and outliers. Correct the MGS price spike, QCK free cash flow mismatch, and BSL profit error.
2. Compare [fundamentals](notebooks/03_fundamental_ratios.ipynb), [financial trends](notebooks/04_financial_trends.ipynb), [price risk](notebooks/05_price_risk_analysis.ipynb), and [valuation](notebooks/06_valuation_analysis.ipynb) in the [company dashboard](notebooks/07_company_comparison.ipynb).
3. [Review market scenarios](notebooks/08_market_briefing.ipynb) and assess how rate changes, public project delays, and energy costs could affect holdings.
4. [Backtest the selection](notebooks/09_factor_backtesting.ipynb). Test the individual factors, then compare basket sizes, stock weights, and holding periods using the combined score.
