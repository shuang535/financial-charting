"""Behavioral checks for the reusable financial charting asset."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "financial_charting.py"
)
SPEC = importlib.util.spec_from_file_location("financial_charting", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load financial charting asset: {MODULE_PATH}")
financial_charting = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = financial_charting
SPEC.loader.exec_module(financial_charting)


def test_plot_two_series_has_no_implicit_source_or_suffix() -> None:
    index = pd.date_range("2025-01-31", periods=4, freq="ME")
    left = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="Left")
    right = pd.Series([4.0, 3.0, 2.0, 1.0], index=index, name="Right")

    figure, _, _ = financial_charting.plot_two_series(left, right)

    try:
        assert figure.texts == []
    finally:
        plt.close(figure)


def test_explicit_source_and_organization_suffix_are_rendered() -> None:
    figure, _ = plt.subplots()

    try:
        financial_charting.add_source_note(
            figure,
            "Bloomberg",
            suffix="Cathay SITE",
        )
        assert [text.get_text() for text in figure.texts] == [
            "Source: Bloomberg, Cathay SITE"
        ]
    finally:
        plt.close(figure)


def test_performance_summary_does_not_mutate_returns() -> None:
    index = pd.date_range("2024-01-31", periods=6, freq="ME")
    returns = pd.DataFrame(
        {
            "Strategy": [0.01, -0.02, 0.03, 0.00, 0.02, -0.01],
            "Benchmark": [0.00, -0.01, 0.02, 0.01, 0.01, -0.02],
        },
        index=index,
    )
    original = returns.copy(deep=True)

    summary = financial_charting.build_performance_summary(returns)

    pd.testing.assert_frame_equal(returns, original)
    assert summary["observations"].eq(6).all()
    assert np.isfinite(summary["total_return"]).all()


def test_chart_style_restores_global_matplotlib_settings() -> None:
    original_linewidth = plt.rcParams["lines.linewidth"]

    with financial_charting.chart_style({"lines.linewidth": 7.0}):
        assert plt.rcParams["lines.linewidth"] == 7.0

    assert plt.rcParams["lines.linewidth"] == original_linewidth
