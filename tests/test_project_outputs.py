
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"


class TestProjectOutputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inventory = pd.read_csv(
            REPORTS / "inventory_optimization_recommendations.csv"
        )

        cls.inventory_summary_df = pd.read_csv(
            REPORTS / "inventory_optimization_summary.csv",
            dtype=str,
        )

        cls.models = pd.read_csv(
            REPORTS / "national_aggregate_model_comparison.csv"
        )

        cls.hierarchy = pd.read_csv(
            REPORTS / "hierarchical_reconciled_wide.csv"
        )

        cls.tft_metrics = pd.read_csv(
            REPORTS / "tft_sample_metrics.csv"
        )

        cls.tft_predictions = pd.read_csv(
            REPORTS / "tft_sample_predictions.csv"
        )

        cls.inventory_summary = dict(
            zip(
                cls.inventory_summary_df["metric"],
                cls.inventory_summary_df["value"],
            )
        )

    def test_dashboard_inputs_exist(self):
        required = [
            REPORTS / "inventory_optimization_summary.csv",
            REPORTS / "inventory_summary_by_store.csv",
            REPORTS / "inventory_optimization_recommendations.csv",
            REPORTS / "national_aggregate_model_comparison.csv",
            REPORTS / "prophet_daily_forecast.csv",
        ]

        for path in required:
            with self.subTest(file=path.name):
                self.assertTrue(path.exists())

    def test_model_metrics_are_valid(self):
        metrics = [
            "WAPE_percent",
            "MAE_units",
            "RMSE_units",
        ]

        self.assertTrue(
            {"Prophet", "LightGBM", "Seasonal Naive"}.issubset(
                set(self.models["model"])
            )
        )

        values = self.models[metrics].to_numpy(dtype=float)

        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= 0).all())

    def test_inventory_demand_balance(self):
        inv = self.inventory

        calculated = (
            inv["expected_fulfilled_units"]
            + inv["expected_lost_sales_units"]
        )

        np.testing.assert_allclose(
            inv["planning_demand_7d"].to_numpy(dtype=float),
            calculated.to_numpy(dtype=float),
            atol=1e-6,
        )

    def test_inventory_stock_balance(self):
        inv = self.inventory

        available = (
            inv["assumed_initial_stock"]
            + inv["recommended_order_units"]
        )

        accounted_for = (
            inv["expected_fulfilled_units"]
            + inv["expected_ending_inventory_units"]
        )

        np.testing.assert_allclose(
            available.to_numpy(dtype=float),
            accounted_for.to_numpy(dtype=float),
            atol=1e-6,
        )

    def test_inventory_values_are_nonnegative(self):
        columns = [
            "planning_demand_7d",
            "assumed_initial_stock",
            "recommended_order_units",
            "expected_fulfilled_units",
            "expected_lost_sales_units",
            "expected_ending_inventory_units",
        ]

        values = self.inventory[columns].to_numpy(dtype=float)

        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= 0).all())

    def test_inventory_respects_budget(self):
        budget = float(
            self.inventory_summary["purchase_budget_limit"]
        )

        purchase_cost = float(
            self.inventory_summary["estimated_purchase_cost"]
        )

        self.assertLessEqual(purchase_cost, budget + 0.01)

    def test_hierarchy_is_consistent(self):
        hierarchy = self.hierarchy

        for state in ["CA", "TX", "WI"]:
            store_cols = [
                col for col in hierarchy.columns
                if col.startswith(state + "_")
                and col.rsplit("_", 1)[-1].isdigit()
            ]

            self.assertGreater(
                len(store_cols), 0,
                f"Toko untuk state {state} tidak ditemukan.",
            )

            state_error = (
                hierarchy[state]
                - hierarchy[store_cols].sum(axis=1)
            ).abs().max()

            self.assertLessEqual(float(state_error), 1e-3)

        national_error = (
            hierarchy["Total"]
            - hierarchy[["CA", "TX", "WI"]].sum(axis=1)
        ).abs().max()

        self.assertLessEqual(float(national_error), 1e-3)

    def test_tft_experiment_outputs(self):
        self.assertTrue(
            {"TFT", "LightGBM", "Seasonal Naive"}.issubset(
                set(self.tft_metrics["model"])
            )
        )

        self.assertEqual(
            self.tft_predictions["series_id"].nunique(), 60
        )

        self.assertEqual(len(self.tft_predictions), 60 * 28)

    
def test_reproducibility_scripts_exist(self):
    required_scripts = [
        "eda.py",
        "baseline_validation.py",
        "lightgbm_validation.py",
        "prophet_validation.py",
        "evaluate_intervals.py",
        "hierarchical_reconciliation.py",
        "inventory_optimization.py",
        "tft_validation.py",
        "validate_project.py",
        "app.py",
    ]

    for filename in required_scripts:
        with self.subTest(script=filename):
            self.assertTrue((ROOT / filename).exists())



if __name__ == "__main__":
    unittest.main(verbosity=2)
