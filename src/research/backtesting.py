"""Point-in-time, price-only research baskets for the fictional universe."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.fundamentals import calculate_fundamentals, latest_fundamentals
from src.research.valuation import valuation_snapshot
from src.research.price_risk import risk_summary


FACTOR_METRICS = {
    'Growth': {'revenue_cagr_3y': True, 'eps_growth': True},
    'Profitability': {'mean_margin_3y': True, 'margin_std_3y': False},
    'Cash generation': {'fcf_margin': True, 'mean_fcf_margin_3y': True},
    'Financial resilience': {'net_debt_to_fcf': False},
    'Valuation': {'earnings_yield': True, 'fcf_yield': True},
    'Low price risk': {'annualised_volatility': False, 'max_drawdown': True},
}
SPECIAL_FINANCIALS = {'CDB', 'BSL', 'CRM'}


def factor_snapshot(annual, prices, as_of):
    """Score equally weighted metric percentiles using only known data.

    Every component must be defined; equal scores get average ranks. Specialist
    financials remain visible but do not enter industrial cash-flow comparisons.
    """
    date = pd.Timestamp(as_of)
    latest = latest_fundamentals(annual, date).set_index('ticker')
    history = calculate_fundamentals(annual.loc[pd.to_datetime(annual.available_date).le(date)])
    recent = history.groupby('ticker', group_keys=False).tail(3)
    averages = recent.groupby('ticker').agg(
        n=('year', 'count'),
        first=('year', 'min'),
        last=('year', 'max'),
        mean_margin_3y=('profit_margin', 'mean'),
        margin_std_3y=('profit_margin', 'std'),
        mean_fcf_margin_3y=('fcf_margin', 'mean'),
        fcf_count=('fcf_margin', 'count'),
    )
    averages.loc[
        ~(averages.n.eq(3) & averages['last'].sub(averages['first']).eq(2)),
        ['mean_margin_3y', 'margin_std_3y', 'mean_fcf_margin_3y'],
    ] = np.nan
    averages.loc[averages.fcf_count.ne(3), 'mean_fcf_margin_3y'] = np.nan
    latest = latest.join(averages[['mean_margin_3y', 'margin_std_3y', 'mean_fcf_margin_3y']])
    valuation = valuation_snapshot(annual, prices, date).set_index('ticker')
    latest = latest.join(valuation[['earnings_yield', 'fcf_yield']])
    trailing = risk_summary(
        prices,
        start=date - pd.DateOffset(years=1),
        end=date,
    ).set_index('ticker')
    latest = latest.join(trailing[['annualised_volatility', 'max_drawdown']])
    records = []
    for factor, metrics in FACTOR_METRICS.items():
        valid = np.isfinite(latest[list(metrics)]).all(axis=1)
        if factor in {'Cash generation', 'Financial resilience', 'Valuation'}:
            valid &= ~latest.index.isin(SPECIAL_FINANCIALS)
        if factor == 'Low price risk':
            valid &= trailing.sessions.reindex(latest.index).ge(200)
        scores = pd.Series(np.nan, index=latest.index, dtype=float)
        if valid.any():
            ranks = [
                latest.loc[valid, metric].rank(
                    method='average', pct=True, ascending=higher
                )
                for metric, higher in metrics.items()
            ]
            scores.loc[valid] = pd.concat(ranks, axis=1).mean(axis=1)
        for ticker, row in latest.iterrows():
            missing = [m for m in metrics if not np.isfinite(row[m])]
            reason = 'Eligible'
            if ticker in SPECIAL_FINANCIALS and factor in {'Cash generation', 'Financial resilience', 'Valuation'}:
                reason = 'Specialist financial: industrial cash-flow/leverage comparison inapplicable'
            elif missing:
                reason = 'Undefined/missing: ' + ', '.join(missing)
            elif not valid.loc[ticker]:
                reason = 'Insufficient trailing prices'
            records.append(dict(
                signal_date=date,
                ticker=ticker,
                company=row.company,
                sector=row.sector,
                factor=factor,
                score=scores.loc[ticker],
                eligible=bool(valid.loc[ticker]),
                reason=reason,
                financial_year=row.year,
                available_date=row.available_date,
                **{m: row[m] for m in metrics},
            ))
    return pd.DataFrame(records)


def _next_close(index, signal):
    later = index[index > pd.Timestamp(signal)]
    return later[0] if len(later) else pd.NaT


def _choose(snapshot, factor, top_n=10):
    valid = snapshot.loc[snapshot.factor.eq(factor) & snapshot.eligible]
    if len(valid) < top_n:
        raise ValueError(f'{factor}: {len(valid)} eligible; need {top_n}')
    return valid.sort_values(['score', 'ticker'], ascending=[False, True]).head(top_n).ticker.tolist()


def _rebalance(weights, target, cost):
    """Fund costs on buys and sells: a = 1 - cost * sum(abs(a*target-current))."""
    a = 1.0
    for _ in range(50):
        updated = 1 - cost * np.abs(a * target - weights).sum()
        if abs(updated - a) < 1e-14:
            a = updated
            break
        a = updated
    return (target.copy(), a, float(np.abs(a * target - weights).sum()))


def simulate_basket(panel, holdings, end_date, cost_bps=0):
    """Equal targets at scheduled closes; holdings drift until the next review.

    Costs include entry and final liquidation. Annual turnover is half buys plus
    sells divided by years, excluding entry and liquidation. No price filling.
    """
    cost = cost_bps / 10000
    dates = panel.loc[min(holdings):end_date].index
    weights = pd.Series(0.0, index=panel.columns)
    wealth, previous, turnover = (1.0, None, 0.0)
    path, periods = ([], [])
    period_base, period_start = (1.0, dates[0])
    for date in dates:
        if previous is not None:
            daily = panel.loc[date] / panel.loc[previous] - 1
            held = weights.gt(0)
            if not np.isfinite(daily.loc[held]).all():
                raise ValueError(f'Missing held price at {date}')
            gross = float((weights.loc[held] * (1 + daily.loc[held])).sum())
            wealth *= gross
            weights.loc[held] *= (1 + daily.loc[held]) / gross
        if date in holdings:
            if previous is not None:
                periods.append(dict(
                    start_date=period_start,
                    end_date=date,
                    **{'return': wealth / period_base - 1},
                ))
                period_base, period_start = (wealth, date)
            target = pd.Series(0.0, index=panel.columns)
            selection = holdings[date]
            if isinstance(selection, dict):
                target.loc[list(selection)] = list(selection.values())
                target_total = target.sum()
                if target_total <= 0 or not np.isfinite(target_total):
                    raise ValueError('Target portfolio weights must be finite and positive')
                target /= target_total
            else:
                target.loc[selection] = 1 / len(selection)
            weights, adjustment, traded = _rebalance(weights, target, cost)
            wealth *= adjustment
            if previous is not None:
                turnover += traded / 2
        if date == end_date:
            wealth *= 1 - cost
            periods.append(dict(
                start_date=period_start,
                end_date=date,
                **{'return': wealth / period_base - 1},
            ))
        path.append((date, wealth))
        previous = date
    nav = pd.Series(dict(path), name='nav').sort_index()
    return (nav, pd.DataFrame(periods), turnover / ((dates[-1] - dates[0]).days / 365.25))


def _summary(nav, turnover, benchmark):
    daily = nav.pct_change(fill_method=None)
    # Include the initial purchase cost in the investor's first daily return.
    daily.iloc[0] = nav.iloc[0] - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    volatility = daily.std(ddof=1) * np.sqrt(252)
    return dict(
        cagr=nav.iloc[-1] ** (1 / years) - 1,
        annualised_volatility=volatility,
        sharpe=(daily.mean() - (1.03 ** (1 / 252) - 1)) / daily.std(ddof=1) * np.sqrt(252) if volatility > 0 else np.nan,
        max_drawdown=(nav / nav.cummax().clip(lower=1) - 1).min(),
        turnover_annualised=turnover,
        benchmark_relative_return=nav.iloc[-1] / benchmark.iloc[-1] - 1,
    )


def run_study(annual, prices):
    """Compare annual and six-month reviews over the common April 2020–April 2025 sample."""
    panel = (
        prices.copy()
        .assign(Date=lambda x: pd.to_datetime(x.Date))
        .set_index('Date')
        .sort_index()
    )
    # Extensions of the price file must respect this study's fixed cutoff.
    panel = panel.loc[panel.index <= pd.Timestamp('2025-12-31')]
    if panel.index.has_duplicates or not np.isfinite(panel.to_numpy()).all() or (panel <= 0).any().any():
        raise ValueError('Require complete, unique and positive approved prices; no filling')
    annual_dates = [pd.Timestamp(y, 3, 31) for y in range(2020, 2025)]
    semi_dates = sorted(annual_dates + [pd.Timestamp(y, 9, 30) for y in range(2020, 2025)])
    end = _next_close(panel.index, '2025-03-31')
    if pd.isna(end):
        raise ValueError('Common exit is unavailable')
    dates = sorted(set(semi_dates + [pd.Timestamp('2025-03-31')]))
    snapshots = {d: factor_snapshot(annual, prices, d) for d in dates}
    selections, paths, summaries, returns, periods, forwards = ([], [], [], [], [], [])
    for schedule, signals in [('Annual', annual_dates), ('Six-month', semi_dates)]:
        for factor in FACTOR_METRICS:
            holdings, eligible_holdings = ({}, {})
            for signal in signals:
                selected = _choose(snapshots[signal], factor)
                execution = _next_close(panel.index, signal)
                holdings[execution] = selected
                rows = snapshots[signal].loc[snapshots[signal].factor.eq(factor)].copy()
                eligible_holdings[execution] = rows.loc[rows.eligible, 'ticker'].tolist()
                rows['selected'] = rows.ticker.isin(selected)
                rows['execution_date'], rows['schedule'] = (execution, schedule)
                selections.append(rows)
            eligible_nav, _, _ = simulate_basket(panel, eligible_holdings, end)
            first = min(holdings)
            benchmark = panel.loc[first:end, 'BENCHMARK'] / panel.loc[first, 'BENCHMARK']
            for cost_bps in [0, 25]:
                nav, period, turnover = simulate_basket(panel, holdings, end, cost_bps)
                key = dict(factor=factor, schedule=schedule, cost_bps=cost_bps)
                summaries.append({
                    **key,
                    **_summary(nav, turnover, benchmark),
                    'eligible_cagr': eligible_nav.iloc[-1] ** (365.25 / (end - first).days) - 1,
                })
                paths.append(pd.DataFrame(dict(
                    Date=nav.index,
                    **key,
                    nav=nav.values,
                    eligible_nav=eligible_nav.values,
                    benchmark_nav=benchmark.values,
                )))
                daily = nav.pct_change(fill_method=None)
                daily.iloc[0] = nav.iloc[0] - 1
                returns.append(pd.DataFrame({'Date': daily.index, **key, 'return': daily.values}))
                for k, v in key.items():
                    period[k] = v
                periods.append(period)
    for signal in annual_dates + [pd.Timestamp('2025-03-31')]:
        entry = _next_close(panel.index, signal)
        for factor in FACTOR_METRICS:
            chosen = _choose(snapshots[signal], factor)
            for months in [6, 12]:
                exit_date = _next_close(panel.index, signal + pd.DateOffset(months=months))
                complete = pd.notna(exit_date)
                ret = (panel.loc[exit_date, chosen] / panel.loc[entry, chosen] - 1).mean() if complete else np.nan
                forwards.append(dict(
                    signal_date=signal,
                    execution_date=entry,
                    factor=factor,
                    horizon_months=months,
                    exit_date=exit_date,
                    complete=complete,
                    **{'return': ret},
                ))
    names = list(FACTOR_METRICS)
    overlap = pd.DataFrame(index=names, columns=names, dtype=float)
    for left in names:
        for right in names:
            overlap.loc[left, right] = np.mean([
                len(
                    set(_choose(snapshots[d], left))
                    & set(_choose(snapshots[d], right))
                ) / 10
                for d in annual_dates
            ])
    return dict(
        summary=pd.DataFrame(summaries),
        nav=pd.concat(paths, ignore_index=True),
        returns=pd.concat(returns, ignore_index=True),
        selections=pd.concat(selections, ignore_index=True),
        forward_returns=pd.DataFrame(forwards),
        period_returns=pd.concat(periods, ignore_index=True),
        overlap=overlap,
        weights=pd.DataFrame(dict(factor=names, weight=[1 / 6] * 6)),
        start_date=_next_close(panel.index, annual_dates[0]),
        end_date=end,
    )


def _blend_paths(study):
    """Compare predefined factor blends with drifting sleeves and annual resets.

    This is a gross research sensitivity, not a fitted equity allocation.
    """
    gross = study['nav'].query("schedule == 'Annual' and cost_bps == 0")
    panel = gross.pivot(index='Date', columns='factor', values='nav')
    review_dates = sorted(study['selections'].query("schedule == 'Annual'").execution_date.unique())
    allocations = {
        'Equal factor weights': {factor: 1 / 6 for factor in FACTOR_METRICS},
        'Cautious sensitivity': {
            'Growth': 0.15,
            'Profitability': 0.15,
            'Cash generation': 0.15,
            'Financial resilience': 0.25,
            'Valuation': 0.2,
            'Low price risk': 0.1,
        },
    }
    records = []
    paths = []
    benchmark = gross.drop_duplicates('Date').set_index('Date').benchmark_nav
    for label, allocation in allocations.items():
        # Factor sleeves drift with prices between annual resets.
        weights = pd.Series(allocation).reindex(panel.columns)
        wealth = 1.0
        previous = None
        rows = []
        for date in panel.index:
            if previous is not None:
                rel = panel.loc[date] / panel.loc[previous]
                multiplier = float((weights * rel).sum())
                wealth *= multiplier
                weights = weights * rel / multiplier
            if date in review_dates:
                weights = pd.Series(allocation).reindex(panel.columns)
            rows.append((date, wealth))
            previous = date
        nav = pd.Series(dict(rows)).sort_index()
        records.append({'blend': label, **_summary(nav, np.nan, benchmark)})
        paths.append(pd.DataFrame({'Date': nav.index, 'blend': label, 'nav': nav.values}))
    return (pd.DataFrame(records), pd.concat(paths, ignore_index=True))


def save_study(study, directory):
    """Export auditable CSVs, standalone figures and a narrative factor-weight report."""
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plot_dir = directory / 'factor_backtest_plots'
    plot_dir.mkdir(exist_ok=True)
    artifacts = {'plots': {}}
    blend_summary, blend_nav = _blend_paths(study)
    study['blend_summary'], study['blend_nav'] = (blend_summary, blend_nav)
    for name in ['summary', 'nav', 'returns', 'selections', 'forward_returns', 'period_returns', 'weights', 'overlap', 'blend_summary', 'blend_nav']:
        path = directory / f'factor_backtest_{name}.csv'
        study[name].to_csv(path, index=name == 'overlap', float_format='%.10g')
        artifacts[name] = path
    annual = study['nav'].query("schedule == 'Annual' and cost_bps == 0")

    def export(title, filename, figure):
        figure.tight_layout()
        path = plot_dir / filename
        figure.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(figure)
        artifacts['plots'][title] = path
    fig, ax = plt.subplots(figsize=(11, 6))
    for factor, group in annual.groupby('factor'):
        ax.plot(group.Date, group.nav * 100, label=factor)
    benchmark = annual.drop_duplicates('Date')
    ax.plot(
        benchmark.Date,
        benchmark.benchmark_nav * 100,
        color='black',
        linestyle='--',
        label='Supplied benchmark',
    )
    ax.set(
        title='Annual factor baskets: growth of £100 (price only, gross)',
        ylabel='Value (£)',
        xlabel='Date',
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    export('Cumulative price growth', '01_cumulative_growth.png', fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for ax, (factor, group) in zip(axes.flat, annual.groupby('factor')):
        ax.plot(group.Date, (group.nav / group.nav.cummax() - 1) * 100, color='#a34c4c')
        ax.set(title=factor, ylabel='Drawdown (%)')
        ax.grid(alpha=0.2)
    fig.suptitle('Annual baskets: declines from preceding price peaks')
    export('Drawdown paths', '02_drawdowns.png', fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    periods = study['period_returns'].query("schedule == 'Annual' and cost_bps == 0").copy()
    periods['Selection year'] = pd.to_datetime(periods.start_date).dt.year
    periods.pivot(
        index='Selection year',
        columns='factor',
        values='return',
    ).mul(100).plot.bar(ax=ax)
    ax.set(
        title='Complete annual holding periods (April to April)',
        ylabel='Price return (%)',
        xlabel='Selection year',
    )
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=0.2)
    export('Returns by annual holding period', '03_period_returns.png', fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    summary = study['summary'].query("schedule == 'Annual' and cost_bps == 0")
    for _, row in summary.iterrows():
        ax.scatter(row.annualised_volatility * 100, row.cagr * 100)
        ax.annotate(
            row.factor,
            (row.annualised_volatility * 100, row.cagr * 100),
            xytext=(4, 4),
            textcoords='offset points',
            fontsize=8,
        )
    ax.set(
        title='Annual factor baskets: observed risk versus return',
        xlabel='Annualised volatility (%)',
        ylabel='CAGR (%)',
    )
    ax.grid(alpha=0.2)
    export('Return versus volatility', '04_risk_return.png', fig)
    returns = study['returns'].query("schedule == 'Annual' and cost_bps == 0").pivot(
        index='Date',
        columns='factor',
        values='return',
    )
    correlations = returns.corr()
    study['factor_correlations'] = correlations
    path = directory / 'factor_backtest_correlations.csv'
    correlations.to_csv(path)
    artifacts['correlations'] = path
    for title, matrix, filename in [('Daily factor-return correlations', correlations, '05_correlations.png'), ('Average top-ten selection overlap', study['overlap'], '06_selection_overlap.png')]:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(matrix, vmin=-1 if 'correlations' in title else 0, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(len(matrix)), matrix.columns, rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                ax.text(j, i, f'{matrix.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
        ax.set_title(title)
        fig.colorbar(
            im,
            ax=ax,
            label='Correlation' if 'correlations' in title else 'Shared selections / 10',
        )
        export(title, filename, fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(axes, ['cagr', 'max_drawdown'], ['CAGR', 'Maximum drawdown']):
        study['summary'].query('cost_bps == 0').pivot(
            index='factor',
            columns='schedule',
            values=metric,
        ).mul(100).plot.barh(ax=ax)
        ax.set(title=title, xlabel='Percent', ylabel='')
        ax.grid(axis='x', alpha=0.2)
    export('Annual versus six-month reviews', '07_rebalance_comparison.png', fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, group in blend_nav.groupby('blend'):
        ax.plot(group.Date, group.nav * 100, label=label)
    ax.set(
        title='Predefined factor-weight sensitivity (gross research sleeves)',
        xlabel='Date',
        ylabel='Value of £100 (price only)',
    )
    ax.legend()
    ax.grid(alpha=0.2)
    export('Factor-weight sensitivity', '08_weight_sensitivity.png', fig)
    # Display totals exactly 100%; calculations retain exact sixths.
    display_weights = [16.67, 16.67, 16.67, 16.67, 16.66, 16.66]
    lines = [
        'FACTOR BACKTEST FINDINGS AND FACTOR-WEIGHT RECOMMENDATION',
        f"Information cutoff: 31 December 2025 | Study: {study['start_date'].date()} to {study['end_date'].date()}",
        'All results are price-only; 2025 figures are provisional estimates.',
        '',
        'METHOD',
        'Signals: 31 March annually; execute at the next available close. Six-month sensitivity also selects after 30 September.',
        'Five complete annual holdings on the common three-year-growth sample; no fine-tuned optimization or claim of statistical significance.',
        'Metrics receive equal percentile-rank weights. All components must be valid; select ten by score, then ticker. Holdings drift between reviews.',
        'Reported net sensitivity charges 25 basis points per buy and sell, including initial entry and final liquidation; benchmark and eligible-universe baselines are gross.',
        'Turnover excludes initial entry and terminal liquidation; half buys plus sells / elapsed years.',
        'Three-year growth and margins use annual records. Six-/twelve-month forward observations are gross individual basket outcomes, not extra independent annual tests.',
        '',
        'SIX FACTOR RESULTS',
    ]
    annual_selections = study['selections'].query("schedule == 'Annual'")
    for factor, metrics in FACTOR_METRICS.items():
        rows = summary.loc[summary.factor.eq(factor)].iloc[0]
        net = study['summary'].query("schedule == 'Annual' and cost_bps == 25").set_index('factor').loc[factor]
        semi = study['summary'].query("schedule == 'Six-month' and cost_bps == 0").set_index('factor').loc[factor]
        eligible = annual_selections.loc[annual_selections.factor.eq(factor)].groupby('signal_date').eligible.sum()
        outcomes = periods.loc[periods.factor.eq(factor), 'return']
        lines += [
            '',
            factor.upper(),
            'Metrics (equal weights): ' + ', '.join((f"{m} ({('higher' if higher else 'lower')} preferred)" for m, higher in metrics.items())),
            f'Eligible companies: {eligible.min()}–{eligible.max()}; top ten selected each year.',
            f'Annual gross CAGR {rows.cagr:.2%}; volatility {rows.annualised_volatility:.2%}; max drawdown {rows.max_drawdown:.2%}; Sharpe {rows.sharpe:.2f}.',
            f'Annual net CAGR {net.cagr:.2%}; semiannual gross CAGR {semi.cagr:.2%}; gross eligible-universe CAGR {rows.eligible_cagr:.2%}.',
            f'Annual turnover {rows.turnover_annualised:.2f}x; positive holdings {outcomes.gt(0).sum()}/{len(outcomes)}; holding-return range {outcomes.min():.2%} to {outcomes.max():.2%}.',
        ]
        reasons = annual_selections.loc[
            annual_selections.factor.eq(factor) & ~annual_selections.eligible,
            ['ticker', 'reason'],
        ].drop_duplicates()
        if not reasons.empty:
            lines += ['Exclusions: ' + '; '.join((f'{r.ticker}: {r.reason}' for r in reasons.itertuples()))]
        sectors = annual_selections.loc[annual_selections.factor.eq(factor) & annual_selections.selected].groupby('sector').size() / 50
        lines += ['Average initial sector weights: ' + ', '.join((f'{s} {w:.0%}' for s, w in sectors.items()))]
    lines += [
        '',
        'FACTOR WEIGHTS — RECOMMEND EQUAL WEIGHTS',
        'The sample is too short and the baskets overlap too much to justify precise performance-fitted tilts. Each factor contributes one-sixth to a future research score; these are not stock or portfolio allocation weights.',
    ]
    for (factor, weight), display in zip(study['weights'].itertuples(index=False, name=None), display_weights):
        lines.append(f'{factor}: {display:.2f}% (exact calculation weight 1/6)')
    lines += [
        'Displayed total: 100.00%. Equal weights apply only when a company has all six valid factor scores; never silently assign zero to missing scores or redistribute weights for specialist financials.',
        'Specialist financials require a separate sector scorecard before a combined score can fairly cover the entire universe.',
        '',
        'PREDEFINED WEIGHT SENSITIVITY — NO WEIGHT SEARCH',
        'Cautious alternative: Growth 15%, Profitability 15%, Cash generation 15%, Financial resilience 25%, Valuation 20%, Low price risk 10%.',
        'Both blends reset factor-sleeve weights annually; sleeve weights drift between resets. These gross mixes illustrate overlapping factor returns, not a fitted stock portfolio.',
        'Blending factor baskets differs from ranking stocks by a combined factor score; no combined-score selection strategy has been backtested here.',
    ]
    for row in blend_summary.itertuples():
        lines.append(f'{row.blend}: CAGR {row.cagr:.2%}, volatility {row.annualised_volatility:.2%}, maximum drawdown {row.max_drawdown:.2%}.')
    lines += [
        '',
        'ECONOMIC FACTORS AND DIVIDENDS',
        'Refinancing and debt resilience: use both debt-reset exposure and latest cash/debt/FCF. Existing loan rates and borrowing spreads are unknown; rate cuts do not guarantee cheaper refinancing.',
        'Discretionary demand: high optional-sales exposure is a downside sensitivity under weak demand and a possible upside exposure if households recover.',
        'Public projects: infrastructure-sales share indicates timing exposure; project delays and payment terms are not quantified.',
        'Essential technology and routine healthcare: supplied sales shares indicate demand exposures, not forecast returns. Budget growth is still positive but slows in the base case.',
        'Energy: disruption risk only; company energy-cost shares and pass-through are unavailable. No quantitative energy ranking is inferred.',
        'All macro management estimates are end-2025 descriptions. Never apply them retrospectively as though they were known in 2020.',
        'Dividends remain a separate yield/payout/FCF sustainability check. Price-only testing cannot measure their total-return benefit; no invented dividend reinvestment is used.',
        '',
        'LIMITATIONS AND PUBLISHED CONTEXT',
        'All five common annual periods are descriptive evidence, not an untouched validation set. No statistical confidence or optimized best weight is claimed.',
        'The supplied 20-company fictional universe is fixed; historical membership/delisting data are unavailable. Results are conditional on this universe.',
        'Growth excludes undefined EPS growth; leverage excludes nonpositive FCF. These eligibility rules can favour existing profitable/cash-generating firms. Eligible-universe benchmarks reveal part of that effect.',
        'Cross-company ranks are not sector neutral. Four names per sector are too few to infer robust within-sector factor effects.',
        'Cash-generation and valuation comparisons exclude CDB/BSL/CRM because one component is structurally unavailable. LNP remains eligible where its inputs are valid.',
        'Metric rank ties average; final score ties sort by ticker. Tiny unrounded numeric differences are retained.',
        'MSCI Quality (2014) averages standardized ROE, debt/equity and earnings-variability scores. That supports a simple equal-combination principle, not our proxy definitions or a claim of index replication.',
        'https://www.msci.com/eqb/methodology/meth_docs/MSCI_Quality_Indexes_Methodology_July_2014.pdf',
        'MSCI diversified multifactor methodology uses different definitions/constraints; its exact weights are not evidence for weights in this exercise.',
        'https://www.msci.com/eqb/methodology/meth_docs/MSCI_Diversified_Multiple-Factor_Indexes_Methodology_May2022.pdf',
        '',
        'See exported selections, component metric values, eligible-universe NAVs, period outcomes and plot files for audit details.',
    ]
    report = directory / 'factor_backtest_findings.txt'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    artifacts['report'] = report
    return artifacts


def composite_snapshot(annual, prices, as_of):
    """Return equal-weight six-factor company scores for fully eligible names.

    Each component is the within-factor percentile score from factor_snapshot;
    a company must have all six component scores. Specialist financials and
    records with undefined factor inputs are excluded explicitly.
    """
    factors = factor_snapshot(annual, prices, as_of)
    scores = factors.pivot(index='ticker', columns='factor', values='score')
    required = list(FACTOR_METRICS)
    eligible = scores[required].notna().all(axis=1)
    metadata = (factors.sort_values('factor').drop_duplicates('ticker').set_index('ticker')
                [['company', 'sector', 'financial_year', 'available_date']])
    result = metadata.join(scores[required])
    trailing_volatility = (factors.loc[factors.factor.eq('Low price risk')]
        .set_index('ticker')['annualised_volatility'])
    result['annualised_volatility'] = trailing_volatility.reindex(result.index)
    result['score'] = scores[required].mean(axis=1, skipna=False)
    result['eligible'] = eligible
    result['reason'] = np.where(eligible, 'Eligible on all six factors', 'At least one factor score undefined')
    result['signal_date'] = pd.Timestamp(as_of)
    return result.reset_index()


def run_composite_study(annual, prices, holding_counts=(5, 8, 10)):
    """Backtest annual equal-factor company scores for several basket sizes.

    These are research baskets equally weighted by pounds. The compare-by-size
    sensitivity informs diversification tradeoffs; it makes no stock allocation.
    """
    panel = (prices.copy().assign(Date=lambda frame: pd.to_datetime(frame.Date))
             .set_index('Date').sort_index())
    panel = panel.loc[:'2025-12-31']
    if (panel.index.has_duplicates or not np.isfinite(panel.to_numpy()).all()
            or (panel <= 0).any().any()):
        raise ValueError('Require complete, unique and positive prices through the cutoff; no filling')
    signal_dates = [pd.Timestamp(year, 3, 31) for year in range(2020, 2025)]
    exit_date = _next_close(panel.index, '2025-03-31')
    snapshots = {date: composite_snapshot(annual, prices, date) for date in signal_dates}
    eligible_counts = [int(snapshot.eligible.sum()) for snapshot in snapshots.values()]
    maximum_supported = min(eligible_counts)
    counts = [count for count in holding_counts if 0 < count <= maximum_supported]
    if len(counts) != len(tuple(holding_counts)):
        raise ValueError(f'Holding counts exceed minimum eligible universe ({maximum_supported})')
    summary, paths, returns, selections, periods = [], [], [], [], []
    for count in counts:
        holdings, eligible_holdings = {}, {}
        for date, snapshot in snapshots.items():
            ranked = snapshot.loc[snapshot.eligible].sort_values(['score', 'ticker'], ascending=[False, True])
            selected = ranked.head(count).ticker.tolist()
            execution = _next_close(panel.index, date)
            holdings[execution] = selected
            eligible_holdings[execution] = snapshot.loc[snapshot.eligible, 'ticker'].tolist()
            ranked = snapshot.copy()
            ranked['selected'] = ranked.ticker.isin(selected)
            ranked['execution_date'] = execution
            ranked['holding_count'] = count
            selections.append(ranked)
        eligible_nav, _, _ = simulate_basket(panel, eligible_holdings, exit_date)
        start = min(holdings)
        benchmark = panel.loc[start:exit_date, 'BENCHMARK'] / panel.loc[start, 'BENCHMARK']
        for cost_bps in (0, 25):
            nav, period, turnover = simulate_basket(panel, holdings, exit_date, cost_bps)
            key = {'factor': 'Equal-factor composite', 'holding_count': count, 'schedule': 'Annual', 'cost_bps': cost_bps}
            summary.append({**key, **_summary(nav, turnover, benchmark),
                            'eligible_cagr': eligible_nav.iloc[-1] ** (365.25 / (exit_date - start).days) - 1})
            paths.append(pd.DataFrame({'Date': nav.index, **key, 'nav': nav.values,
                                      'eligible_nav': eligible_nav.values, 'benchmark_nav': benchmark.values}))
            daily = nav.pct_change(fill_method=None)
            daily.iloc[0] = nav.iloc[0] - 1
            returns.append(pd.DataFrame({'Date': daily.index, **key, 'return': daily.values}))
            period['holding_count'], period['cost_bps'] = count, cost_bps
            periods.append(period)
    return {'summary': pd.DataFrame(summary), 'nav': pd.concat(paths, ignore_index=True),
            'returns': pd.concat(returns, ignore_index=True), 'selections': pd.concat(selections, ignore_index=True),
            'period_returns': pd.concat(periods, ignore_index=True), 'snapshots': pd.concat(snapshots.values(), ignore_index=True),
            'start_date': _next_close(panel.index, signal_dates[0]), 'end_date': exit_date,
            'holding_counts': counts}


def run_volatility_tilt_study(annual, prices, holding_count=5, budget_gbp=100_000):
    """Compare equal pounds with inverse-trailing-volatility weights.

    Both methods choose the top composite-score names and rebalance annually.
    Inverse volatility uses the prior 12 months at each signal date.
    """
    panel = (prices.copy().assign(Date=lambda frame: pd.to_datetime(frame.Date))
             .set_index('Date').sort_index())
    panel = panel.loc[:'2025-12-31']
    if (panel.index.has_duplicates or not np.isfinite(panel.to_numpy()).all()
            or (panel <= 0).any().any()):
        raise ValueError('Require complete, unique and positive prices through the cutoff; no filling')
    signal_dates = [pd.Timestamp(year, 3, 31) for year in range(2020, 2025)]
    exit_date = _next_close(panel.index, '2025-03-31')
    holdings_by_method = {'Equal pounds': {}, 'Inverse-volatility tilt': {}}
    selection_records = []
    for signal in signal_dates:
        snapshot = composite_snapshot(annual, prices, signal)
        eligible = snapshot.loc[snapshot.eligible].copy()
        if not eligible.annualised_volatility.gt(0).all():
            raise ValueError(f'Invalid trailing volatility at {signal.date()}')
        chosen = eligible.sort_values(['score', 'ticker'], ascending=[False, True]).head(holding_count).copy()
        execution = _next_close(panel.index, signal)
        equal_weights = {ticker: 1 / holding_count for ticker in chosen.ticker}
        inverse = 1 / chosen.annualised_volatility
        tilted_weights = pd.Series(
            inverse.to_numpy() / inverse.sum(), index=chosen.ticker
        ).to_dict()
        holdings_by_method['Equal pounds'][execution] = equal_weights
        holdings_by_method['Inverse-volatility tilt'][execution] = tilted_weights
        chosen['signal_date'] = signal
        chosen['execution_date'] = execution
        chosen['holding_count'] = holding_count
        chosen['equal_weight'] = chosen.ticker.map(equal_weights)
        chosen['inverse_volatility_weight'] = chosen.ticker.map(tilted_weights)
        selection_records.append(chosen)
    current = composite_snapshot(annual, prices, '2025-12-31')
    candidates = current.loc[current.eligible].sort_values(['score', 'ticker'], ascending=[False, True]).head(holding_count).copy()
    inverse = 1 / candidates.annualised_volatility
    candidates['equal_weight'] = 1 / holding_count
    candidates['inverse_volatility_weight'] = inverse.to_numpy() / inverse.sum()
    candidates['share_price_gbp'] = candidates.ticker.map(panel.loc[pd.Timestamp('2025-12-31')])
    candidates['equal_budget_gbp'] = budget_gbp * candidates.equal_weight
    candidates['tilted_budget_gbp'] = budget_gbp * candidates.inverse_volatility_weight
    candidates['equal_whole_shares'] = np.floor(candidates.equal_budget_gbp / candidates.share_price_gbp).astype(int)
    candidates['tilted_whole_shares'] = np.floor(candidates.tilted_budget_gbp / candidates.share_price_gbp).astype(int)
    candidates['equal_residual_cash_gbp'] = candidates.equal_budget_gbp - candidates.equal_whole_shares * candidates.share_price_gbp
    candidates['tilted_residual_cash_gbp'] = candidates.tilted_budget_gbp - candidates.tilted_whole_shares * candidates.share_price_gbp
    first = min(min(holdings) for holdings in holdings_by_method.values())
    benchmark = panel.loc[first:exit_date, 'BENCHMARK'] / panel.loc[first, 'BENCHMARK']
    summaries, paths, period_frames = [], [], []
    for method, holdings in holdings_by_method.items():
        for cost_bps in (0, 25):
            nav, periods, turnover = simulate_basket(panel, holdings, exit_date, cost_bps)
            summaries.append({'weighting': method, 'holding_count': holding_count, 'cost_bps': cost_bps,
                              **_summary(nav, turnover, benchmark)})
            paths.append(pd.DataFrame({'Date': nav.index, 'weighting': method, 'cost_bps': cost_bps,
                                       'nav': nav.values, 'benchmark_nav': benchmark.values}))
            periods['weighting'], periods['cost_bps'] = method, cost_bps
            period_frames.append(periods)
    return {'summary': pd.DataFrame(summaries), 'nav': pd.concat(paths, ignore_index=True),
            'period_returns': pd.concat(period_frames, ignore_index=True),
            'selections': pd.concat(selection_records, ignore_index=True),
            'current_positions': candidates, 'start_date': first, 'end_date': exit_date}


def run_three_year_study(annual, prices):
    """Compare fixed holdings and annual reselection in overlapping three-year windows."""
    panel = prices.assign(Date=lambda x: pd.to_datetime(x.Date)).set_index('Date').sort_index().loc[:'2025-12-31']
    if panel.index.has_duplicates or not np.isfinite(panel.to_numpy()).all() or (panel <= 0).any().any():
        raise ValueError('Require unique, complete positive prices; no filling')
    summaries, paths, selections = [], [], []
    for year in (2020, 2021, 2022):
        exit_date = _next_close(panel.index, pd.Timestamp(year + 3, 3, 31))
        if pd.isna(exit_date):
            raise ValueError(f'Incomplete three-year window starting {year}')
        for method in ('Buy and hold', 'Annual reselection'):
            holdings = {}
            for review_year in ([year] if method == 'Buy and hold' else range(year, year + 3)):
                signal = pd.Timestamp(review_year, 3, 31)
                snapshot = composite_snapshot(annual, prices, signal)
                ranked = snapshot.loc[snapshot.eligible].sort_values(['score', 'ticker'], ascending=[False, True])
                if len(ranked) < 5:
                    raise ValueError('Fewer than five eligible companies')
                execution = _next_close(panel.index, signal)
                chosen = ranked.head(5).ticker.tolist()
                holdings[execution] = chosen
                snapshot = snapshot.assign(window_start=year, method=method, execution_date=execution,
                                           selected=snapshot.ticker.isin(chosen))
                selections.append(snapshot)
            start = min(holdings)
            benchmark = panel.loc[start:exit_date, 'BENCHMARK'] / panel.loc[start, 'BENCHMARK']
            for cost in (0, 25):
                nav, _, turnover = simulate_basket(panel, holdings, exit_date, cost)
                key = dict(window_start=year, method=method, cost_bps=cost)
                summaries.append(dict(**key, start_date=start, end_date=exit_date,
                    cumulative_return=nav.iloc[-1]-1, ending_value_gbp=100000*nav.iloc[-1],
                    benchmark_return=benchmark.iloc[-1]-1, **_summary(nav, turnover, benchmark)))
                paths.append(pd.DataFrame(dict(Date=nav.index, nav=nav.values,
                    benchmark_nav=benchmark.values, **key)))
    return dict(summary=pd.DataFrame(summaries), nav=pd.concat(paths, ignore_index=True),
                selections=pd.concat(selections, ignore_index=True))
