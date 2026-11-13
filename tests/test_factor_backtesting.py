"""Small numerical and information-timing checks for the factor experiments."""
import unittest

import numpy as np
import pandas as pd

from src.data_loader import load_cleaned_data
from src.research.backtesting import factor_snapshot, simulate_basket, run_study


class BasketAccountingTests(unittest.TestCase):

    def test_weights_drift_between_reviews(self):
        dates = pd.date_range('2020-01-01', periods=3)
        panel = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]}, index=dates)
        nav, periods, _ = simulate_basket(panel, {dates[0]: ['A', 'B']}, dates[-1])
        self.assertAlmostEqual(nav.iloc[-1], 2.0)
        self.assertAlmostEqual((1 + periods['return']).prod(), nav.iloc[-1])

    def test_costs_are_self_financed_on_both_sides(self):
        dates = pd.date_range('2020-01-01', periods=3)
        panel = pd.DataFrame({'A': [1, 1, 1]}, index=dates)
        nav, _, turnover = simulate_basket(panel, {dates[0]: ['A']}, dates[-1], 25)
        self.assertAlmostEqual(nav.iloc[-1], (1 - 0.0025) / (1 + 0.0025), places=12)
        self.assertEqual(turnover, 0)

    def test_rebalance_and_period_chain(self):
        dates = pd.date_range('2020-01-01', periods=4)
        panel = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 1, 2, 3]}, index=dates)
        nav, periods, _ = simulate_basket(panel, {dates[0]: ['A'], dates[2]: ['B']}, dates[-1], 25)
        expected = 3 * (1 - 0.0025) ** 2 / (1 + 0.0025) ** 2 * 1.5
        self.assertAlmostEqual(nav.iloc[-1], expected, places=10)
        self.assertAlmostEqual((1 + periods['return']).prod(), nav.iloc[-1], places=12)


class InformationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.annual, cls.prices = load_cleaned_data('data/Cleaned')

    def test_future_observations_do_not_change_signals(self):
        date = pd.Timestamp('2022-03-31')
        original = factor_snapshot(self.annual, self.prices, date)
        annual = self.annual.copy()
        prices = self.prices.copy()
        annual.loc[annual.available_date.gt(date), 'profit_gbp_m'] *= 1000
        prices.loc[prices.Date.gt(date), prices.columns.drop('Date')] *= 10
        changed = factor_snapshot(annual, prices, date)
        pd.testing.assert_frame_equal(original, changed)
        self.assertTrue(original.available_date.le(date).all())

    def test_study_eligibility_timing_and_costs(self):
        # A future extension must not complete the March 2025 twelve-month test.
        future_prices = self.prices.tail(1).copy()
        future_prices['Date'] = pd.Timestamp('2026-04-01')
        extended_prices = pd.concat([self.prices, future_prices], ignore_index=True)
        study = run_study(self.annual, extended_prices)
        selected = study['selections'].query('selected')
        self.assertTrue(selected.groupby(['schedule', 'signal_date', 'factor']).size().eq(10).all())
        self.assertTrue(selected.execution_date.gt(selected.signal_date).all())
        self.assertEqual(study['start_date'], pd.Timestamp('2020-04-01'))
        self.assertEqual(study['end_date'], pd.Timestamp('2025-04-01'))
        periods = study['period_returns'].groupby(['factor', 'schedule', 'cost_bps']).size()
        self.assertTrue(periods.xs('Annual', level='schedule').eq(5).all())
        self.assertTrue(periods.xs('Six-month', level='schedule').eq(10).all())
        summary = study['summary']
        numeric = summary.select_dtypes('number')
        self.assertTrue(np.isfinite(numeric).all().all())
        compare = summary.pivot(index=['factor', 'schedule'], columns='cost_bps', values='cagr')
        self.assertTrue(compare[25].lt(compare[0]).all())
        self.assertEqual((~study['forward_returns'].complete).sum(), 6)
        self.assertAlmostEqual(study['weights'].weight.sum(), 1)


if __name__ == '__main__':
    unittest.main()
