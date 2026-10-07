"""Behavioral checks of chart inputs, geometry and report-size delivery.

Output-layer tests use synthetic prepared analysis results. Direct Matplotlib
examples exercise the contracts that do not require a toolkit helper.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image

from test_financial_charting import financial_charting as charts


@pytest.fixture(autouse=True)
def report_style() -> Iterator[None]:
    """Restore style and close figures even when an assertion fails."""
    with charts.chart_style({"font.size": 8, "legend.fontsize": 7}):
        yield
    plt.close("all")


def assert_text_layout(figure: Figure, texts: list[Text]) -> None:
    """Check named visible labels for canvas clipping and pairwise overlap."""
    figure.set_dpi(300)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer) for text in texts if text.get_text()]
    canvas = figure.bbox
    for box in boxes:
        assert box.x0 >= canvas.x0 and box.x1 <= canvas.x1
        assert box.y0 >= canvas.y0 and box.y1 <= canvas.y1
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            assert not first.overlaps(second)


def test_unknown_future_outcome_is_only_an_x_reference(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-31", periods=18, freq="ME")
    x = pd.Series(np.arange(18.0), index=dates)
    y = pd.Series(0.3 * x.to_numpy() + np.sin(x), index=dates)
    y.iloc[-3:] = np.nan  # Prepared future outcomes not yet observed.
    originals = (x.copy(), y.copy())

    fig, ax, stats = charts.plot_regression(
        x, y, show_stats=False, figsize=(4.2, 3.4),
        title="最新環境與未來三個月變化\n示意資料；未實現結果不納入估計",
        x_label="已知指標", y_label="未來三個月變化（百分點）",
    )
    latest_x = float(x.iloc[-1])
    ax.axvline(latest_x, color="gray", linestyle=":")
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 2, 4])
    ax.annotate("2025/06", xy=(latest_x, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -26), textcoords="offset points", ha="center",
                annotation_clip=False)
    ax.text(0.03, 0.94, f"有效 N={stats.n_obs}\n未來結果尚未觀察",
            transform=ax.transAxes, va="top")
    charts.add_source_note(fig, "Synthetic data", fontsize=7)
    fig.subplots_adjust(left=0.17, right=0.93, top=0.80, bottom=0.29)
    ax.xaxis.labelpad = 22

    points = ax.collections[0].get_offsets()
    assert stats.n_obs == 15 and len(points) == 15
    assert latest_x not in points[:, 0]
    pd.testing.assert_series_equal(x, originals[0])
    pd.testing.assert_series_equal(y, originals[1])
    assert_text_layout(fig, [ax.title, ax.xaxis.label, ax.yaxis.label,
                             *ax.texts, *fig.texts,
                             *ax.get_xticklabels(), *ax.get_yticklabels()])
    charts.save_figure(fig, tmp_path / "unknown_future_y.png", bbox_inches=None)
    charts.save_figure(fig, tmp_path / "unknown_future_y_preview.png", dpi=96, bbox_inches=None)


@pytest.mark.parametrize("left_kind", ["line", "bar"])
def test_mixed_axes_preserve_layers_units_and_endpoint_bars(
    tmp_path: Path, left_kind: Literal["line", "bar"],
) -> None:
    dates = pd.date_range("2025-01-31", periods=8, freq="ME")
    left = pd.Series([180, 200, np.nan, 240, 190, 160, 175, 150], index=dates)
    right = pd.Series([5.4, 5.6, 5.8, 6.0, 5.7, 5.2, 5.4, 5.0], index=dates)
    right_kind = "bar" if left_kind == "line" else "line"
    colors = ("#002060", "#BFBFBF") if left_kind == "line" else ("#BFBFBF", "#002060")

    fig, left_ax, right_ax = charts.plot_two_series(
        left, right, secondary_y=True, kinds=(left_kind, right_kind),
        colors=colors, bar_width_days=16, labels=("OAS", "YTW"),
        y_labels=("OAS（基點）", "YTW（%）"),
        title="信用利差與殖利率\n線／柱雙軸示意", figsize=(4.2, 3.4),
        legend_location="upper center", legend_ncol=2,
    )
    line_ax, bar_ax = (left_ax, right_ax) if left_kind == "line" else (right_ax, left_ax)
    assert right_ax is not None
    # Leave headroom for the legend and darken the gray axis for small print.
    left_ax.set_ylim(140 if left_kind == "line" else 0, 290)
    right_ax.set_ylim(0, 8)
    bar_ax.tick_params(axis="y", colors="#6B7280")
    bar_ax.yaxis.label.set_color("#6B7280")
    left_ax.set_xticks(dates[::2])
    left_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    assert line_ax.get_zorder() > bar_ax.get_zorder()
    assert not line_ax.patch.get_visible()
    assert bar_ax.patches and all(patch.get_visible() for patch in bar_ax.patches)
    assert len(bar_ax.patches) == (8 if left_kind == "line" else 7)
    for patch in bar_ax.patches:
        assert patch.get_width() == 16
        assert left_ax.get_xlim()[0] <= patch.get_x()
        assert patch.get_x() + patch.get_width() <= left_ax.get_xlim()[1]
    assert left_ax.get_ylabel() == "OAS（基點）"
    assert right_ax.get_ylabel() == "YTW（%）"
    assert line_ax.get_legend() is not None
    assert set(text.get_text() for text in line_ax.get_legend().get_texts()) == {"OAS", "YTW (RHS)"}
    charts.save_figure(fig, tmp_path / f"mixed_{left_kind}.png", bbox_inches=None)


@pytest.mark.parametrize("values", [([-4, 2, 7], [-80, 5, 10]), ([1, 2, 7], [10, 20, 30]),
                                    ([-7, -2, -1], [-30, -20, -10]), ([0, 0, 0], [0, 0, 0])])
def test_zero_alignment_preserves_all_data_and_prior_limits(
    values: tuple[list[float], list[float]], tmp_path: Path,
) -> None:
    dates = pd.date_range("2025-01-31", periods=3, freq="ME")
    fig, left_ax, right_ax = charts.plot_two_series(
        pd.Series(values[0], index=dates), pd.Series(values[1], index=dates),
        secondary_y=True, y_labels=("指標 A（%）", "指標 B（基點）"),
        title="零點對齊；兩側振幅不可直接比較", figsize=(5, 3.4),
    )
    assert right_ax is not None
    old_limits = [axis.get_ylim() for axis in (left_ax, right_ax)]

    charts.align_zero_axes(left_ax, right_ax)

    for axis, old, data in zip((left_ax, right_ax), old_limits, values):
        low, high = axis.get_ylim()
        assert low <= min(old[0], min(data)) and high >= max(old[1], max(data))
        axis.axhline(0, color="gray", linewidth=0.8)
    fig.canvas.draw()
    assert left_ax.transData.transform((0, 0))[1] == pytest.approx(right_ax.transData.transform((0, 0))[1])
    charts.save_figure(fig, tmp_path / "zero_alignment.png", bbox_inches=None)


def test_zero_alignment_rejects_unsupported_axes_without_partial_mutation() -> None:
    fig, left_ax = plt.subplots()
    right_ax = left_ax.twinx()
    right_ax.set_yscale("log")
    before = left_ax.get_ylim()
    with pytest.raises(ValueError, match="linear"):
        charts.align_zero_axes(left_ax, right_ax)
    assert left_ax.get_ylim() == before


def test_sign_fill_breaks_at_nan_and_meets_zero_at_crossings(tmp_path: Path) -> None:
    dates = pd.date_range("2025-01-31", periods=9, freq="ME")
    values = np.array([2., 1., -1., -2., np.nan, -2., 0., 2., 1.])
    lower = np.zeros(9)
    valid = np.isfinite(values) & np.isfinite(lower)
    top = np.ma.masked_where(~valid, values)
    bottom = np.ma.masked_where(~valid, lower)
    positive_mask, negative_mask = values > 0, values < 0
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(dates, values, color="#002060", label="政策利率變化")
    fills = [
        ax.fill_between(dates, bottom, top, where=valid & positive_mask,
                        color="#C83E4D", interpolate=True, label="升息"),
        ax.fill_between(dates, bottom, top, where=valid & negative_mask,
                        color="#83BC5C", interpolate=True, label="降息"),
    ]
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("政策變化相對零線填色\n五月缺值；零值不屬於升降息")
    ax.set_ylabel("百分點")
    ax.legend(loc="lower left")
    charts.configure_time_axis(ax)
    ax.set_xticks(dates[::2])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    fig.tight_layout()

    gap = mdates.date2num(dates[4])
    crossing = (mdates.date2num(dates[1]) + mdates.date2num(dates[2])) / 2
    for fill, sign in zip(fills, (1, -1)):
        assert fill.get_paths()
        vertices = np.concatenate([path.vertices for path in fill.get_paths()])
        assert np.any(np.isclose(vertices[:, 0], crossing, atol=1e-8, rtol=0))
        assert np.all(vertices[:, 1] * sign >= -1e-12)
        for path in fill.get_paths():
            assert not path.vertices[:, 0].min() < gap < path.vertices[:, 0].max()
    charts.save_figure(fig, tmp_path / "fill_with_gap.png", bbox_inches=None)


def test_horizons_report_their_own_effective_group_counts(tmp_path: Path) -> None:
    dates = pd.date_range("2023-01-31", periods=24, freq="ME")
    x = pd.Series(np.arange(24.), index=dates)
    groups = pd.Series(np.where(np.arange(24) % 2, "tightening", "other"), index=dates)
    figures: list[Figure] = []
    observed_n: list[int] = []
    for horizon in (1, 3, 6, 12):
        y = pd.Series(np.sin(x) + 0.2 * x, index=dates)
        y.iloc[-horizon:] = np.nan  # Prepared synthetic horizon data.
        fig, ax, stats = charts.plot_grouped_regression(
            x, y, groups, group_colors={"tightening": "#C83E4D", "other": "#7A8793"},
            group_labels={"tightening": "模型緊縮區間", "other": "其他區間"},
            show_stats=False, figsize=(4.2, 3.4),
            title=f"未來 {horizon} 個月變化；有效 N={24 - horizon}",
            x_label="指標", y_label="變化（百分點）",
        )
        figures.append(fig)
        observed_n.append(sum(stat.n_obs for stat in stats.values()))
        counts = groups.loc[y.notna()].value_counts()
        assert {group: stat.n_obs for group, stat in stats.items()} == counts.to_dict()
        assert len(ax.collections) == 2
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert all("n=" in label and "R²" not in label for label in labels)
        charts.save_figure(fig, tmp_path / f"horizon_{horizon}.png", bbox_inches=None)
    assert observed_n == [23, 21, 18, 12]
    assert len({id(fig) for fig in figures}) == 4


def test_time_series_gaps_latest_dates_and_input_immutability() -> None:
    dates = pd.date_range("2025-01-31", periods=6, freq="ME")
    left = pd.Series([1, np.nan, 3, 4, np.nan, np.nan], index=dates)
    right = pd.Series([np.nan, 6, 5, np.nan, 3, np.nan], index=dates)
    original = left.copy()

    _, ax, _ = charts.plot_two_series(left, right)

    assert np.isnan(ax.lines[0].get_ydata()[1])
    assert np.isnan(ax.lines[1].get_ydata()[3])
    assert ax.get_xlim() == pytest.approx(mdates.date2num([dates[0], dates[4]]))
    pd.testing.assert_series_equal(left, original)


def test_return_diagnostics_reject_missing_returns() -> None:
    dates = pd.date_range("2025-01-31", periods=5, freq="ME")
    data = pd.Series([0.01, np.nan, -0.02, 0.03, 0.01], index=dates)
    with pytest.raises(ValueError, match="missing"):
        charts.plot_underperformance_diagnostics(data, data, rolling_window=2)


def test_small_canvas_source_legend_latest_and_precomputed_quantiles(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-31", periods=18, freq="ME")
    values = pd.Series(90 + np.arange(18) + 4 * np.sin(np.arange(18)), index=dates)
    # The analysis fixture computes results before they reach the renderer.
    levels = {f"P{percentile}": float(values.quantile(percentile / 100))
              for percentile in (20, 50, 80)}
    latest_value = float(values.iloc[-1])
    supplied_rank = float(values.le(latest_value).mean() * 100)
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    ax.plot(dates, values, color="#002060", label="示意指標")
    for label, value in levels.items():
        ax.axhline(value, color="#7A8793", linestyle="--", linewidth=0.8,
                   label=f"{label} = {value:.1f}")
    ax.set_title("歷史分位位置與最新觀察\n固定樣本；示意資料")
    ax.set_ylabel("指標（點）")
    ax.set_ylim(85, 120)
    ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=40))
    ax.scatter([dates[-1]], [latest_value], facecolors="none", edgecolors="#002060")
    ax.text(0.97, 0.94, f"2025/06：{latest_value:.1f}\n歷史排名 {supplied_rank:.0f}%",
            transform=ax.transAxes, ha="right", va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    legend = ax.legend(loc="lower left", ncol=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    charts.add_source_note(
        fig, "Synthetic data\n固定樣本 2024/01–2025/06；月頻；分位數：線性插值\n"
        "排名：小於等於當期值比例，含當期；由分析層提供", fontsize=7,
    )
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.23, top=0.81)
    texts = [ax.title, ax.yaxis.label, *ax.texts, *legend.get_texts(),
             *fig.texts, *ax.get_xticklabels(), *ax.get_yticklabels()]

    assert_text_layout(fig, texts)
    for line, value in zip(ax.lines[1:], levels.values()):
        assert list(line.get_ydata()) == [value, value]
    output = charts.save_figure(fig, tmp_path / "small_quantiles.png", bbox_inches=None)
    with Image.open(output) as image:
        assert image.size == (1320, 1050)
    charts.save_figure(fig, tmp_path / "small_quantiles_preview.png", dpi=96, bbox_inches=None)


def test_hidden_statistics_leave_estimates_unchanged() -> None:
    x = pd.Series(np.arange(8.0))
    y = pd.Series(2 * x + np.sin(x))
    _, full_ax, full_stats = charts.plot_regression(x, y)
    _, compact_ax, compact_stats = charts.plot_regression(x, y, show_stats=False)
    assert full_stats == compact_stats
    assert len(full_ax.texts) == 1 and len(compact_ax.texts) == 0


def test_seasonal_current_year_preserves_interior_gap() -> None:
    dates = pd.date_range("2022-01-31", periods=42, freq="ME")
    values = pd.Series(100 + np.sin(np.arange(42)), index=dates)
    values.iloc[-4] = np.nan
    _, ax = charts.plot_seasonality(values)
    current_line = next(line for line in ax.lines if line.get_label() == "2025 (YTD)")
    assert np.isnan(current_line.get_ydata()[2])
    assert len(current_line.get_xdata()) == 12


def test_annual_sum_does_not_turn_missing_year_into_zero() -> None:
    dates = pd.date_range("2023-01-31", periods=36, freq="ME")
    values = pd.Series(1.0, index=dates)
    values.iloc[12:24] = np.nan
    _, ax = charts.plot_annual_bars(values, aggregation="sum")
    assert np.isnan(ax.patches[1].get_height())


def test_fixed_canvas_overrides_tight_style(tmp_path: Path) -> None:
    with plt.rc_context({"savefig.bbox": "tight"}):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("A compact chart")
        output = charts.save_figure(fig, tmp_path / "fixed_canvas.png", bbox_inches=None)
        assert plt.rcParams["savefig.bbox"] == "tight"
    with Image.open(output) as image:
        assert image.size == (1200, 900)
