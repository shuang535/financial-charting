"""Output-layer helpers for reproducible financial research charts.

The module is import-safe: it performs no I/O and changes no global style until
``apply_chart_style`` is called explicitly.
"""

from __future__ import annotations

import calendar
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Literal, Mapping, Sequence, TypeAlias

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cycler, rc_context, rcParams
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter

BRAND_COLORS: tuple[str, ...] = (
    "#002060",
    "#18C1FF",
    "#83BC5C",
    "#FFC000",
    "#F672A7",
    "#EF8B47",
    "#C198E0",
    "#BFBFBF",
    "#343A40",
)

DateLike: TypeAlias = str | date | datetime | pd.Timestamp
Period: TypeAlias = tuple[DateLike, DateLike]
Limits: TypeAlias = tuple[float | None, float | None]


@dataclass(frozen=True)
class RegressionStats:
    """Summary returned with a regression chart."""

    intercept: float
    slope: float
    r_squared: float
    t_stat_slope: float
    n_obs: int
    quadratic_coefficient: float | None = None


def _style_settings() -> dict[str, object]:
    return {
        "axes.prop_cycle": cycler(color=BRAND_COLORS),
        "axes.grid": False,
        "axes.unicode_minus": False,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft JhengHei",
            "Noto Sans CJK TC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.handlelength": 2,
        "lines.linewidth": 2,
        "savefig.dpi": 300,
    }


def apply_chart_style(overrides: Mapping[str, object] | None = None) -> None:
    """Explicitly apply the chart style to the current Python process."""
    settings = _style_settings()
    if overrides:
        settings.update(overrides)
    rcParams.update(settings)


@contextmanager
def chart_style(
    overrides: Mapping[str, object] | None = None,
) -> Iterator[None]:
    """Temporarily apply the chart style without permanent global mutation."""
    settings = _style_settings()
    if overrides:
        settings.update(overrides)
    with rc_context(rc=settings):
        yield


def _prepare_time_series(series: pd.Series, argument_name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{argument_name} must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"{argument_name}.index must be a DatetimeIndex")
    if series.index.has_duplicates:
        raise ValueError(f"{argument_name}.index contains duplicate timestamps")

    prepared = pd.to_numeric(series.copy(), errors="coerce").sort_index().dropna()
    if prepared.empty:
        raise ValueError(f"{argument_name} has no numeric observations")
    return prepared.astype(float)


def _slice_series(
    series: pd.Series,
    start: DateLike | None,
    end: DateLike | None,
) -> pd.Series:
    lower = pd.Timestamp(start) if start is not None else series.index.min()
    upper = pd.Timestamp(end) if end is not None else series.index.max()
    sliced = series.loc[lower:upper]
    if sliced.empty:
        raise ValueError("The selected date range has no observations")
    return sliced


def configure_time_axis(ax: Axes, year_step: int | None = None) -> None:
    """Configure readable date ticks for short or long samples."""
    if year_step is not None:
        if year_step < 1:
            raise ValueError("year_step must be at least 1")
        locator = mdates.YearLocator(base=year_step)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelrotation=45)


def shade_periods(
    ax: Axes,
    periods: Sequence[Period],
    *,
    color: str = "lightgray",
    alpha: float = 0.3,
    label: str | None = None,
) -> None:
    """Shade event periods and add at most one legend label."""
    for index, (start, end) in enumerate(periods):
        start_timestamp = pd.Timestamp(start)
        end_timestamp = pd.Timestamp(end)
        if start_timestamp > end_timestamp:
            raise ValueError(f"Period start {start!r} is after end {end!r}")
        span_label = label if index == 0 and label is not None else "_nolegend_"
        ax.axvspan(
            start_timestamp,
            end_timestamp,
            color=color,
            alpha=alpha,
            label=span_label,
            zorder=0,
        )


def merge_legends(
    primary_ax: Axes,
    secondary_ax: Axes | None = None,
    *,
    ncol: int = 1,
    location: str = "upper left",
) -> None:
    """Merge handles from one or two axes without duplicate labels."""
    axes = (primary_ax,) if secondary_ax is None else (primary_ax, secondary_ax)
    unique: dict[str, object] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label != "_nolegend_" and label not in unique:
                unique[label] = handle
    if unique:
        primary_ax.legend(
            list(unique.values()),
            list(unique.keys()),
            loc=location,
            ncol=ncol,
        )


def add_source_note(
    fig: Figure,
    source: str,
    *,
    suffix: str | None = None,
    fontsize: float = 9,
) -> None:
    """Add a source note to a specific figure."""
    text = f"Source: {source}"
    if suffix:
        text = f"{text}, {suffix}"
    fig.text(0.01, 0.01, text, fontsize=fontsize, ha="left", va="bottom")


def _finish_figure(fig: Figure, has_source: bool) -> None:
    bottom = 0.07 if has_source else 0.02
    fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))


def plot_two_series(
    left: pd.Series,
    right: pd.Series,
    *,
    secondary_y: bool = False,
    labels: tuple[str, str] | None = None,
    title: str | None = None,
    source: str | None = None,
    start: DateLike | None = None,
    end: DateLike | None = None,
    left_ylim: Limits | None = None,
    right_ylim: Limits | None = None,
    periods: Sequence[Period] = (),
    period_label: str | None = "Recession",
    year_step: int | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Axes | None]:
    """Plot two financial time series on one axis or two explicit axes."""
    left_data = _slice_series(_prepare_time_series(left, "left"), start, end)
    right_data = _slice_series(_prepare_time_series(right, "right"), start, end)

    if ax is None:
        fig, primary_ax = plt.subplots(figsize=figsize)
    else:
        primary_ax = ax
        fig = ax.figure
    secondary_ax = primary_ax.twinx() if secondary_y else None
    right_ax = secondary_ax or primary_ax

    if labels is None:
        labels = (
            str(left.name) if left.name is not None else "Left series",
            str(right.name) if right.name is not None else "Right series",
        )
    right_label = f"{labels[1]} (RHS)" if secondary_y else labels[1]

    primary_ax.plot(left_data.index, left_data, color=BRAND_COLORS[0], label=labels[0])
    right_ax.plot(right_data.index, right_data, color=BRAND_COLORS[1], label=right_label)
    if secondary_y:
        primary_ax.tick_params(axis="y", colors=BRAND_COLORS[0])
    if secondary_ax is not None:
        secondary_ax.tick_params(axis="y", colors=BRAND_COLORS[1])

    if left_ylim is not None:
        primary_ax.set_ylim(*left_ylim)
    if right_ylim is not None:
        right_ax.set_ylim(*right_ylim)
    if periods:
        shade_periods(primary_ax, periods, label=period_label)

    lower = min(left_data.index.min(), right_data.index.min())
    upper = max(left_data.index.max(), right_data.index.max())
    primary_ax.set_xlim(lower, upper)
    primary_ax.set_title(title or "")
    primary_ax.grid(False)
    configure_time_axis(primary_ax, year_step)
    merge_legends(primary_ax, secondary_ax)
    if source:
        add_source_note(fig, source)
    _finish_figure(fig, source is not None)
    return fig, primary_ax, secondary_ax


def plot_seasonality(
    series: pd.Series,
    *,
    baseline_period: Period | None = None,
    recent_complete_years: int = 3,
    include_current_year: bool = True,
    band_std: float = 1.0,
    title: str | None = None,
    source: str | None = None,
    month_labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot monthly seasonality with a descriptive mean ± standard deviation band."""
    if recent_complete_years < 1:
        raise ValueError("recent_complete_years must be at least 1")
    if band_std < 0:
        raise ValueError("band_std cannot be negative")

    data = _prepare_time_series(series, "series")
    year_month = data.index.to_period("M")
    if year_month.has_duplicates:
        raise ValueError("series must contain at most one observation per year-month")

    frame = pd.DataFrame(
        {"year": data.index.year, "month": data.index.month, "value": data.values}
    )
    pivot = frame.pivot(index="month", columns="year", values="value").reindex(range(1, 13))

    if baseline_period is None:
        baseline = pivot
    else:
        baseline_data = _slice_series(data, baseline_period[0], baseline_period[1])
        baseline_frame = pd.DataFrame(
            {
                "year": baseline_data.index.year,
                "month": baseline_data.index.month,
                "value": baseline_data.values,
            }
        )
        baseline = baseline_frame.pivot(
            index="month", columns="year", values="value"
        ).reindex(range(1, 13))

    insufficient_months = baseline.count(axis=1).loc[lambda count: count < 2].index.tolist()
    if insufficient_months:
        raise ValueError(
            "Baseline needs at least two observations for every month; "
            f"insufficient months: {insufficient_months}"
        )

    mean = baseline.mean(axis=1)
    std = baseline.std(axis=1)
    current_year = int(data.index.max().year)
    complete_years = [int(year) for year in pivot.columns if pivot[year].count() == 12]
    history_years = [year for year in complete_years if year != current_year]
    history_years = history_years[-recent_complete_years:]

    if ax is None:
        fig, chart_ax = plt.subplots(figsize=figsize)
    else:
        chart_ax = ax
        fig = ax.figure

    months = np.arange(1, 13)
    chart_ax.fill_between(
        months,
        mean - band_std * std,
        mean + band_std * std,
        color="lightgray",
        alpha=0.45,
        label=f"Mean ± {band_std:g} SD",
        zorder=0,
    )
    chart_ax.plot(months, mean, color="#808080", linestyle="--", label="Mean")
    for index, year in enumerate(history_years):
        chart_ax.plot(
            months,
            pivot[year],
            marker="o",
            color=BRAND_COLORS[index % len(BRAND_COLORS)],
            label=str(year),
        )

    if include_current_year and current_year in pivot.columns:
        current = pivot[current_year].dropna()
        if not current.empty and current_year not in history_years:
            chart_ax.plot(
                current.index,
                current.values,
                marker="o",
                linewidth=3,
                color="#EF4444",
                label=f"{current_year} (YTD)" if len(current) < 12 else str(current_year),
            )

    labels = list(month_labels) if month_labels is not None else list(calendar.month_abbr[1:])
    if len(labels) != 12:
        raise ValueError("month_labels must contain exactly 12 labels")
    chart_ax.set_xticks(months, labels)
    chart_ax.set_xlim(1, 12)
    chart_ax.set_title(title or (str(series.name) if series.name is not None else "Seasonality"))
    chart_ax.grid(False)
    merge_legends(chart_ax, ncol=2)
    if source:
        add_source_note(fig, source)
    _finish_figure(fig, source is not None)
    return fig, chart_ax


def _prepare_regression_data(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
        raise TypeError("x and y must be pandas Series")
    frame = pd.concat(
        [pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")],
        axis=1,
        join="inner",
    ).dropna()
    frame.columns = ["x", "y"]
    if len(frame) < 3:
        raise ValueError("At least three aligned numeric observations are required")
    return frame.astype(float)


def _estimate_ols(frame: pd.DataFrame, quadratic: bool) -> RegressionStats:
    x_values = frame["x"].to_numpy()
    y_values = frame["y"].to_numpy()
    columns = [np.ones(len(frame)), x_values]
    if quadratic:
        columns.append(x_values**2)
    design = np.column_stack(columns)
    coefficients, _, rank, _ = np.linalg.lstsq(design, y_values, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("Regression design is singular")

    fitted = design @ coefficients
    residual_sum = float(np.sum((y_values - fitted) ** 2))
    total_sum = float(np.sum((y_values - y_values.mean()) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    degrees_of_freedom = len(frame) - design.shape[1]
    if degrees_of_freedom <= 0:
        raise ValueError("Not enough observations for the requested regression")
    covariance = residual_sum / degrees_of_freedom * np.linalg.inv(design.T @ design)
    slope_standard_error = float(np.sqrt(covariance[1, 1]))
    t_stat_slope = (
        float(coefficients[1] / slope_standard_error)
        if slope_standard_error > 0
        else float("nan")
    )
    quadratic_coefficient = float(coefficients[2]) if quadratic else None
    return RegressionStats(
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        r_squared=r_squared,
        t_stat_slope=t_stat_slope,
        n_obs=len(frame),
        quadratic_coefficient=quadratic_coefficient,
    )


def plot_regression(
    x: pd.Series,
    y: pd.Series,
    *,
    quadratic: bool = False,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, RegressionStats]:
    """Plot aligned observations and an OLS fit on a sorted x grid."""
    frame = _prepare_regression_data(x, y)
    stats = _estimate_ols(frame, quadratic)

    if ax is None:
        fig, chart_ax = plt.subplots(figsize=figsize)
    else:
        chart_ax = ax
        fig = ax.figure

    chart_ax.scatter(frame["x"], frame["y"], color=BRAND_COLORS[0], alpha=0.7)
    x_grid = np.linspace(frame["x"].min(), frame["x"].max(), 200)
    fitted = stats.intercept + stats.slope * x_grid
    if stats.quadratic_coefficient is not None:
        fitted = fitted + stats.quadratic_coefficient * x_grid**2
        equation = (
            f"y = {stats.intercept:.2f} + {stats.slope:.2f}x "
            f"+ {stats.quadratic_coefficient:.2f}x²"
        )
    else:
        equation = f"y = {stats.intercept:.2f} + {stats.slope:.2f}x"
    chart_ax.plot(x_grid, fitted, color="#EF4444", linestyle="--", linewidth=2.5)

    annotation = (
        f"{equation}\nR² = {stats.r_squared:.3f}\n"
        f"t(slope) = {stats.t_stat_slope:.2f}\nn = {stats.n_obs}"
    )
    chart_ax.text(
        0.03,
        0.97,
        annotation,
        transform=chart_ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    chart_ax.set_xlabel(x_label or (str(x.name) if x.name is not None else "x"))
    chart_ax.set_ylabel(y_label or (str(y.name) if y.name is not None else "y"))
    chart_ax.set_title(title or "")
    chart_ax.grid(False)
    _finish_figure(fig, False)
    return fig, chart_ax, stats


def plot_grouped_regression(
    x: pd.Series,
    y: pd.Series,
    groups: pd.Series,
    *,
    group_colors: Mapping[str, str],
    group_labels: Mapping[str, str] | None = None,
    quadratic: bool = False,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    show_stats: bool = True,
    figsize: tuple[float, float] = (6.0, 6.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, dict[str, RegressionStats]]:
    """Plot separate scatter points and OLS fits for observed groups."""
    if not isinstance(groups, pd.Series):
        raise TypeError("groups must be a pandas Series")
    if groups.index.has_duplicates:
        raise ValueError("groups.index contains duplicate labels")
    frame = _prepare_regression_data(x, y)
    aligned_groups = groups.reindex(frame.index)
    if aligned_groups.isna().any():
        raise ValueError("groups must cover every aligned x and y observation")
    frame = frame.assign(group=aligned_groups)
    frame["group"] = frame["group"].astype(str)
    observed_groups = list(dict.fromkeys(frame["group"]))
    missing_colors = sorted(set(observed_groups).difference(group_colors))
    if missing_colors:
        raise ValueError(f"Missing colors for groups: {missing_colors}")

    if ax is None:
        fig, chart_ax = plt.subplots(figsize=figsize)
    else:
        chart_ax = ax
        fig = ax.figure

    statistics: dict[str, RegressionStats] = {}
    for group in observed_groups:
        group_frame = frame.loc[frame["group"].eq(group), ["x", "y"]]
        if len(group_frame) < 3:
            raise ValueError(f"Group {group!r} needs at least three observations")
        stats = _estimate_ols(group_frame, quadratic)
        statistics[group] = stats
        color = group_colors[group]
        label = _display_name(group, group_labels)
        if show_stats:
            label = f"{label} (R²={stats.r_squared:.2f}, n={stats.n_obs})"
        chart_ax.scatter(
            group_frame["x"],
            group_frame["y"],
            color=color,
            alpha=0.65,
            label=label,
        )
        x_grid = np.linspace(group_frame["x"].min(), group_frame["x"].max(), 200)
        fitted = stats.intercept + stats.slope * x_grid
        if stats.quadratic_coefficient is not None:
            fitted = fitted + stats.quadratic_coefficient * x_grid**2
        chart_ax.plot(x_grid, fitted, color=color, linewidth=2.5)

    chart_ax.axhline(0.0, color="#7A8793", linewidth=0.8, alpha=0.6)
    chart_ax.set_xlabel(x_label or (str(x.name) if x.name is not None else "x"))
    chart_ax.set_ylabel(y_label or (str(y.name) if y.name is not None else "y"))
    chart_ax.set_title(title or "")
    chart_ax.grid(alpha=0.18)
    merge_legends(chart_ax)
    _finish_figure(fig, False)
    return fig, chart_ax, statistics


def plot_annual_bars(
    series: pd.Series,
    *,
    aggregation: Literal["mean", "sum", "last"] = "mean",
    title: str | None = None,
    positive_color: str = "#002060",
    negative_color: str = "#B22222",
    source: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Aggregate a time series by year and color bars by sign."""
    data = _prepare_time_series(series, "series")
    grouped = data.groupby(data.index.year)
    if aggregation == "mean":
        annual = grouped.mean()
    elif aggregation == "sum":
        annual = grouped.sum()
    else:
        annual = grouped.last()

    if ax is None:
        fig, chart_ax = plt.subplots(figsize=figsize)
    else:
        chart_ax = ax
        fig = ax.figure
    colors = [positive_color if value >= 0 else negative_color for value in annual]
    chart_ax.bar(annual.index.astype(str), annual.values, color=colors)
    chart_ax.axhline(0, color="#343A40", linewidth=0.8)
    chart_ax.set_title(title or (str(series.name) if series.name is not None else "Annual values"))
    chart_ax.tick_params(axis="x", labelrotation=45)
    chart_ax.grid(axis="y", alpha=0.2)
    if source:
        add_source_note(fig, source)
    _finish_figure(fig, source is not None)
    return fig, chart_ax


def _prepare_numeric_frame(
    frame: pd.DataFrame,
    argument_name: str,
    *,
    require_complete: bool,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{argument_name} must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{argument_name}.index must be a DatetimeIndex")
    if frame.empty:
        raise ValueError(f"{argument_name} cannot be empty")
    if frame.index.has_duplicates:
        raise ValueError(f"{argument_name}.index contains duplicate timestamps")
    if frame.columns.has_duplicates:
        raise ValueError(f"{argument_name}.columns contains duplicates")
    if not all(isinstance(column, str) for column in frame.columns):
        raise TypeError(f"{argument_name}.columns must contain strings")

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    invalid = frame.notna() & numeric.isna()
    if invalid.any(axis=None):
        raise ValueError(f"{argument_name} contains non-numeric values")
    if require_complete and numeric.isna().any(axis=None):
        raise ValueError(
            f"{argument_name} contains missing values; align the common sample first"
        )
    if numeric.notna().sum().sum() == 0:
        raise ValueError(f"{argument_name} has no numeric observations")
    return numeric.astype(float).sort_index()


def _validate_periods_per_year(periods_per_year: int) -> None:
    if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool):
        raise TypeError("periods_per_year must be an integer")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")


def _drawdown_from_wealth(
    wealth: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    # WHY: clipping at the initial wealth of 1 captures a loss in the first period.
    running_peak = wealth.cummax().clip(lower=1.0)
    return wealth.div(running_peak) - 1.0


def _annualized_weight_change_turnover(
    target_weights: pd.DataFrame,
    *,
    return_index: pd.DatetimeIndex,
    periods_per_year: int,
    turnover_sides: Literal["one_way", "two_way"],
) -> float:
    weights = _prepare_numeric_frame(
        target_weights,
        "target_weights",
        require_complete=True,
    )
    missing_dates = return_index.difference(weights.index)
    if not missing_dates.empty:
        raise ValueError("target_weights must cover the complete return sample")
    aligned = weights.reindex(return_index)
    period_turnover = aligned.diff().abs().sum(axis=1, min_count=1).dropna()
    if period_turnover.empty:
        return float("nan")
    side_factor = 0.5 if turnover_sides == "one_way" else 1.0
    return float(period_turnover.mean() * periods_per_year * side_factor)


def build_performance_summary(
    period_returns: pd.DataFrame,
    *,
    periods_per_year: int = 12,
    annual_risk_free_rate: float = 0.0,
    target_weights: Mapping[str, pd.DataFrame] | None = None,
    turnover_sides: Literal["one_way", "two_way"] = "one_way",
) -> pd.DataFrame:
    """Build a performance table for arbitrary assets or strategies."""
    _validate_periods_per_year(periods_per_year)
    returns = _prepare_numeric_frame(
        period_returns,
        "period_returns",
        require_complete=True,
    )
    if len(returns) < 2:
        raise ValueError("period_returns needs at least two observations")
    if returns.le(-1.0).any(axis=None):
        raise ValueError("returns cannot be less than or equal to -100%")
    if not np.isfinite(annual_risk_free_rate) or annual_risk_free_rate <= -1.0:
        raise ValueError("annual_risk_free_rate must be finite and greater than -100%")
    if turnover_sides not in {"one_way", "two_way"}:
        raise ValueError("turnover_sides must be 'one_way' or 'two_way'")

    wealth = (1.0 + returns).cumprod()
    total_return = wealth.iloc[-1] - 1.0
    cagr = wealth.iloc[-1].pow(periods_per_year / len(returns)) - 1.0
    annualized_volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    periodic_risk_free_rate = (
        (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    )
    sharpe = (
        (returns.mean() - periodic_risk_free_rate)
        * np.sqrt(periods_per_year)
        / returns.std(ddof=1).replace(0.0, np.nan)
    )
    max_drawdown = _drawdown_from_wealth(wealth).min()
    calmar = cagr / max_drawdown.abs().replace(0.0, np.nan)
    positive_period_probability = returns.gt(0.0).mean()

    turnover_column = f"annual_turnover_{turnover_sides}"
    annual_turnover = pd.Series(np.nan, index=returns.columns, dtype=float)
    if target_weights is not None:
        if not isinstance(target_weights, Mapping):
            raise TypeError("target_weights must map return columns to weight DataFrames")
        unknown_portfolios = sorted(set(target_weights).difference(returns.columns))
        if unknown_portfolios:
            raise ValueError(
                f"target_weights contains unknown return columns: {unknown_portfolios}"
            )
        for portfolio, weights in target_weights.items():
            annual_turnover.loc[portfolio] = _annualized_weight_change_turnover(
                weights,
                return_index=returns.index,
                periods_per_year=periods_per_year,
                turnover_sides=turnover_sides,
            )

    summary = pd.DataFrame(
        {
            "total_return": total_return,
            "cagr": cagr,
            "annualized_volatility": annualized_volatility,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "positive_period_probability": positive_period_probability,
            turnover_column: annual_turnover,
            "observations": len(returns),
        }
    )
    summary.index.name = "series"
    return summary


def build_annual_performance_table(
    period_returns: pd.DataFrame,
    *,
    benchmark: str | None = None,
    periods_per_year: int = 12,
    include_partial_years: bool = False,
    minimum_periods: int | None = None,
    include_excess: bool = True,
) -> pd.DataFrame:
    """Compound periodic returns into calendar-year returns."""
    _validate_periods_per_year(periods_per_year)
    returns = _prepare_numeric_frame(
        period_returns,
        "period_returns",
        require_complete=True,
    )
    if returns.le(-1.0).any(axis=None):
        raise ValueError("returns cannot be less than or equal to -100%")
    if benchmark is not None and benchmark not in returns.columns:
        raise ValueError(f"benchmark {benchmark!r} is not a return column")
    required_periods = periods_per_year if minimum_periods is None else minimum_periods
    if not isinstance(required_periods, int) or isinstance(required_periods, bool):
        raise TypeError("minimum_periods must be an integer")
    if required_periods <= 0:
        raise ValueError("minimum_periods must be positive")

    grouped = returns.groupby(returns.index.year)
    annual = grouped.apply(lambda frame: (1.0 + frame).prod() - 1.0)
    if not include_partial_years:
        complete_years = grouped.size().loc[lambda counts: counts >= required_periods].index
        annual = annual.reindex(complete_years)
    if annual.empty:
        raise ValueError("no calendar years satisfy the requested sample rule")
    annual.index.name = "year"

    if benchmark is not None and include_excess:
        for column in returns.columns:
            if column != benchmark:
                annual[f"{column}_excess_vs_{benchmark}"] = (
                    annual[column] - annual[benchmark]
                )
    return annual


def build_benchmark_win_summary(
    period_returns: pd.DataFrame,
    *,
    benchmark: str,
    periods_per_year: int = 12,
    include_partial_years: bool = False,
    minimum_periods: int | None = None,
) -> pd.DataFrame:
    """Summarize calendar-year wins and losses against a benchmark."""
    annual = build_annual_performance_table(
        period_returns,
        benchmark=benchmark,
        periods_per_year=periods_per_year,
        include_partial_years=include_partial_years,
        minimum_periods=minimum_periods,
        include_excess=False,
    )
    challengers = [column for column in annual.columns if column != benchmark]
    if not challengers:
        raise ValueError("period_returns needs at least one non-benchmark column")

    records: list[dict[str, object]] = []
    down_market = annual[benchmark].lt(0.0)
    for column in challengers:
        excess = annual[column] - annual[benchmark]
        wins = excess.gt(0.0)
        lagging_years = ", ".join(str(year) for year in excess.index[excess.lt(0.0)])
        records.append(
            {
                "series": column,
                "years": len(excess),
                "wins": int(wins.sum()),
                "win_rate": float(wins.mean()),
                "mean_excess_return": float(excess.mean()),
                "median_excess_return": float(excess.median()),
                "best_year": int(excess.idxmax()),
                "best_excess_return": float(excess.max()),
                "worst_year": int(excess.idxmin()),
                "worst_excess_return": float(excess.min()),
                "down_market_years": int(down_market.sum()),
                "down_market_wins": int((wins & down_market).sum()),
                "lagging_years": lagging_years,
            }
        )
    return pd.DataFrame.from_records(records).set_index("series")


def build_regime_return_table(
    period_returns: pd.DataFrame,
    regimes: pd.Series,
    *,
    statistic: Literal["mean", "median", "compound", "annualized"] = "mean",
    periods_per_year: int = 12,
    regime_groups: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Summarize arbitrary asset or strategy returns by regime."""
    _validate_periods_per_year(periods_per_year)
    returns = _prepare_numeric_frame(
        period_returns,
        "period_returns",
        require_complete=True,
    )
    if returns.le(-1.0).any(axis=None):
        raise ValueError("returns cannot be less than or equal to -100%")
    if statistic not in {"mean", "median", "compound", "annualized"}:
        raise ValueError("unsupported regime statistic")
    if not isinstance(regimes, pd.Series):
        raise TypeError("regimes must be a pandas Series")
    if not isinstance(regimes.index, pd.DatetimeIndex):
        raise TypeError("regimes.index must be a DatetimeIndex")
    if regimes.index.has_duplicates:
        raise ValueError("regimes.index contains duplicate timestamps")
    missing_dates = returns.index.difference(regimes.index)
    if not missing_dates.empty:
        raise ValueError("regimes must cover the complete return sample")
    aligned_regimes = regimes.reindex(returns.index)
    if aligned_regimes.isna().any():
        raise ValueError("regimes cannot contain missing values in the return sample")
    aligned_regimes = aligned_regimes.astype(str)

    if regime_groups is None:
        observed = list(dict.fromkeys(aligned_regimes.tolist()))
        groups: Mapping[str, Sequence[str]] = {
            state: (state,) for state in observed
        }
    else:
        if not isinstance(regime_groups, Mapping) or not regime_groups:
            raise ValueError("regime_groups must be a non-empty mapping")
        groups = regime_groups

    rows: dict[str, pd.Series] = {}
    counts: dict[str, int] = {}
    for label, members in groups.items():
        if not isinstance(label, str):
            raise TypeError("regime_groups keys must be strings")
        member_states = tuple(str(member) for member in members)
        if not member_states:
            raise ValueError(f"regime group {label!r} cannot be empty")
        group_returns = returns.loc[aligned_regimes.isin(member_states)]
        counts[label] = len(group_returns)
        if group_returns.empty:
            rows[label] = pd.Series(np.nan, index=returns.columns, dtype=float)
        elif statistic == "mean":
            rows[label] = group_returns.mean()
        elif statistic == "median":
            rows[label] = group_returns.median()
        elif statistic == "compound":
            rows[label] = (1.0 + group_returns).prod() - 1.0
        else:
            rows[label] = (1.0 + group_returns).prod().pow(
                periods_per_year / len(group_returns)
            ) - 1.0

    table = pd.DataFrame.from_dict(rows, orient="index")
    table["sample_n"] = pd.Series(counts)
    table.index.name = "regime"
    return table


def build_drawdown_episodes(period_returns: pd.DataFrame) -> pd.DataFrame:
    """Return peak, trough, recovery and duration for every drawdown episode."""
    returns = _prepare_numeric_frame(
        period_returns,
        "period_returns",
        require_complete=True,
    )
    if returns.le(-1.0).any(axis=None):
        raise ValueError("returns cannot be less than or equal to -100%")

    records: list[dict[str, object]] = []
    tolerance = 1e-12
    for column in returns.columns:
        wealth = (1.0 + returns[column]).cumprod()
        drawdown = _drawdown_from_wealth(wealth)
        start_position: int | None = None

        for position, value in enumerate(drawdown):
            if value < -tolerance and start_position is None:
                start_position = position
            elif value >= -tolerance and start_position is not None:
                end_position = position - 1
                records.append(
                    _summarize_drawdown_episode(
                        column,
                        wealth,
                        drawdown,
                        start_position,
                        end_position,
                        recovery_position=position,
                    )
                )
                start_position = None

        if start_position is not None:
            records.append(
                _summarize_drawdown_episode(
                    column,
                    wealth,
                    drawdown,
                    start_position,
                    len(drawdown) - 1,
                    recovery_position=None,
                )
            )

    columns = [
        "series",
        "peak_date",
        "start_date",
        "trough_date",
        "recovery_date",
        "max_drawdown",
        "periods_to_trough",
        "periods_to_recovery",
        "underwater_periods",
        "recovered",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records, columns=columns)


def _summarize_drawdown_episode(
    series_name: str,
    wealth: pd.Series,
    drawdown: pd.Series,
    start_position: int,
    end_position: int,
    *,
    recovery_position: int | None,
) -> dict[str, object]:
    episode = drawdown.iloc[start_position : end_position + 1]
    trough_date = episode.idxmin()
    trough_position = int(drawdown.index.get_loc(trough_date))
    prior_wealth = wealth.iloc[:start_position]
    if not prior_wealth.empty and prior_wealth.max() >= 1.0:
        peak_date = prior_wealth.loc[prior_wealth.eq(prior_wealth.max())].index[-1]
    else:
        peak_date = pd.NaT
    recovery_date = (
        drawdown.index[recovery_position]
        if recovery_position is not None
        else pd.NaT
    )
    periods_to_recovery = (
        recovery_position - trough_position
        if recovery_position is not None
        else np.nan
    )
    return {
        "series": series_name,
        "peak_date": peak_date,
        "start_date": drawdown.index[start_position],
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "max_drawdown": float(episode.min()),
        "periods_to_trough": trough_position - start_position + 1,
        "periods_to_recovery": periods_to_recovery,
        "underwater_periods": end_position - start_position + 1,
        "recovered": recovery_position is not None,
    }


def _prepare_flag_frame(
    flags: pd.DataFrame | None,
    *,
    index: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    if flags is None:
        return pd.DataFrame(False, index=index, columns=columns)
    if not isinstance(flags, pd.DataFrame):
        raise TypeError("flags must be a pandas DataFrame")
    if not isinstance(flags.index, pd.DatetimeIndex):
        raise TypeError("flags.index must be a DatetimeIndex")
    if flags.index.has_duplicates or flags.columns.has_duplicates:
        raise ValueError("flags cannot contain duplicate index or columns")
    missing_dates = index.difference(flags.index)
    missing_columns = columns.difference(flags.columns)
    if not missing_dates.empty or not missing_columns.empty:
        raise ValueError("flags must cover every plotted date and column")
    aligned = flags.reindex(index=index, columns=columns)
    if aligned.isna().any(axis=None):
        raise ValueError("flags cannot contain missing values")
    if not aligned.isin([True, False]).all(axis=None):
        raise ValueError("flags must contain boolean values")
    return aligned.astype(bool)


def _prepare_boolean_series(
    flags: pd.Series,
    *,
    index: pd.DatetimeIndex,
    argument_name: str,
) -> pd.Series:
    if not isinstance(flags, pd.Series):
        raise TypeError(f"{argument_name} must be a pandas Series")
    if not isinstance(flags.index, pd.DatetimeIndex):
        raise TypeError(f"{argument_name}.index must be a DatetimeIndex")
    if flags.index.has_duplicates:
        raise ValueError(f"{argument_name}.index contains duplicate timestamps")
    missing_dates = index.difference(flags.index)
    if not missing_dates.empty:
        raise ValueError(f"{argument_name} must cover every plotted date")
    aligned = flags.reindex(index)
    if aligned.isna().any() or not aligned.isin([True, False]).all():
        raise ValueError(f"{argument_name} must contain complete boolean values")
    return aligned.astype(bool)


def _display_name(
    name: str,
    display_names: Mapping[str, str] | None,
) -> str:
    if display_names is None:
        return name
    return display_names.get(name, name)


def _shade_monthly_states(
    axes: Sequence[Axes],
    states: pd.Series,
    state_colors: Mapping[str, str],
    *,
    alpha: float,
) -> None:
    if not isinstance(states, pd.Series):
        raise TypeError("regimes must be a pandas Series")
    if not isinstance(states.index, pd.DatetimeIndex):
        raise TypeError("regimes.index must be a DatetimeIndex")
    if states.index.has_duplicates:
        raise ValueError("regimes.index contains duplicate timestamps")
    if states.isna().any():
        raise ValueError("regimes cannot contain missing values")

    for timestamp, raw_state in states.sort_index().items():
        state = str(raw_state)
        if state not in state_colors:
            raise ValueError(f"No color supplied for regime {state!r}")
        start = timestamp.to_period("M").start_time
        end = timestamp.to_period("M").end_time
        for axis in axes:
            axis.axvspan(
                start,
                end,
                color=state_colors[state],
                alpha=alpha,
                linewidth=0,
                zorder=0,
            )


def plot_cumulative_relative_returns(
    monthly_returns: pd.DataFrame,
    *,
    benchmark: str,
    title: str | None = None,
    display_names: Mapping[str, str] | None = None,
    series_colors: Mapping[str, str] | None = None,
    regimes: pd.Series | None = None,
    regime_colors: Mapping[str, str] | None = None,
    regime_alpha: float = 0.08,
    source: str | None = None,
    figsize: tuple[float, float] = (12.0, 7.0),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot cumulative returns and cumulative wealth relative to a benchmark.

    Args:
        monthly_returns: Complete common-sample decimal monthly returns.
        benchmark: Column used as the denominator of relative wealth.
        title: Optional figure title.
        display_names: Optional mapping from input columns to legend labels.
        series_colors: Optional mapping from input columns to line colors.
        regimes: Optional monthly state series used for background shading.
        regime_colors: Required state-to-color mapping when regimes is provided.
        regime_alpha: Transparency of regime background shading.
        source: Optional source note.
        figsize: Figure size in inches.

    Returns:
        The figure and a tuple of cumulative-return and relative-return axes.

    Raises:
        ValueError: If the common sample is incomplete or the benchmark is absent.
    """
    returns = _prepare_numeric_frame(
        monthly_returns,
        "monthly_returns",
        require_complete=True,
    )
    if benchmark not in returns.columns:
        raise ValueError(f"benchmark {benchmark!r} is not a return column")
    if len(returns.columns) < 2:
        raise ValueError("monthly_returns needs at least two columns")
    if returns.le(-1.0).any(axis=None):
        raise ValueError("monthly returns cannot be less than or equal to -100%")
    if not 0.0 <= regime_alpha <= 1.0:
        raise ValueError("regime_alpha must be between 0 and 1")

    wealth = (1.0 + returns).cumprod()
    cumulative_returns = wealth - 1.0
    relative_returns = wealth.div(wealth[benchmark], axis=0) - 1.0

    figure, axes_array = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    cumulative_ax, relative_ax = axes_array
    axes = (cumulative_ax, relative_ax)

    if regimes is not None:
        if regime_colors is None:
            raise ValueError("regime_colors is required when regimes is supplied")
        aligned_regimes = regimes.reindex(returns.index)
        if aligned_regimes.isna().any():
            raise ValueError("regimes must cover the complete return sample")
        _shade_monthly_states(
            axes,
            aligned_regimes,
            regime_colors,
            alpha=regime_alpha,
        )

    for index, column in enumerate(returns.columns):
        is_benchmark = column == benchmark
        if series_colors is not None and column in series_colors:
            color = series_colors[column]
        elif is_benchmark:
            color = "#7A8793"
        else:
            color = BRAND_COLORS[index % len(BRAND_COLORS)]
        cumulative_ax.plot(
            cumulative_returns.index,
            cumulative_returns[column],
            color=color,
            linewidth=2.2 if is_benchmark else 3.0,
            linestyle="--" if is_benchmark else "-",
            label=_display_name(column, display_names),
            zorder=3,
        )
        if not is_benchmark:
            relative_ax.plot(
                relative_returns.index,
                relative_returns[column],
                color=color,
                linewidth=2.5,
                label=(
                    f"{_display_name(column, display_names)} / "
                    f"{_display_name(benchmark, display_names)}"
                ),
                zorder=3,
            )

    cumulative_ax.set_title(title or "")
    cumulative_ax.set_ylabel("Cumulative return")
    relative_ax.set_ylabel("Relative return")
    relative_ax.axhline(0.0, color="#7A8793", linewidth=1.0, zorder=2)
    for axis in axes:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", alpha=0.18)
        axis.set_xlim(returns.index.min(), returns.index.max())
    configure_time_axis(relative_ax)
    merge_legends(cumulative_ax, ncol=min(3, len(returns.columns)))
    merge_legends(relative_ax, ncol=min(2, len(returns.columns) - 1))
    if source:
        add_source_note(figure, source)
    _finish_figure(figure, source is not None)
    return figure, axes


def plot_backtest_diagnostics(
    monthly_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    benchmark: str | None = None,
    title: str | None = None,
    display_names: Mapping[str, str] | None = None,
    series_colors: Mapping[str, str] | None = None,
    weight_colors: Mapping[str, str] | None = None,
    source: str | None = None,
    figsize: tuple[float, float] = (12.0, 9.0),
) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """Plot cumulative wealth, drawdowns and portfolio target weights."""
    returns = _prepare_numeric_frame(
        monthly_returns,
        "monthly_returns",
        require_complete=True,
    )
    weights = _prepare_numeric_frame(
        target_weights,
        "target_weights",
        require_complete=True,
    )
    if benchmark is not None and benchmark not in returns.columns:
        raise ValueError(f"benchmark {benchmark!r} is not a return column")
    missing_weight_dates = returns.index.difference(weights.index)
    if not missing_weight_dates.empty:
        raise ValueError("target_weights must cover the complete return sample")
    weights = weights.reindex(returns.index)
    if returns.le(-1.0).any(axis=None):
        raise ValueError("monthly returns cannot be less than or equal to -100%")

    wealth = (1.0 + returns).cumprod()
    drawdowns = _drawdown_from_wealth(wealth)
    figure, axes_array = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.1, 1.1)},
    )
    wealth_ax, drawdown_ax, weights_ax = axes_array
    axes = (wealth_ax, drawdown_ax, weights_ax)

    for index, column in enumerate(returns.columns):
        is_benchmark = column == benchmark
        if series_colors is not None and column in series_colors:
            color = series_colors[column]
        elif is_benchmark:
            color = "#7A8793"
        else:
            color = BRAND_COLORS[index % len(BRAND_COLORS)]
        line_style = "--" if is_benchmark else "-"
        line_width = 2.0 if is_benchmark else 2.7
        label = _display_name(column, display_names)
        wealth_ax.plot(
            wealth.index,
            wealth[column],
            color=color,
            linestyle=line_style,
            linewidth=line_width,
            label=label,
        )
        drawdown_ax.plot(
            drawdowns.index,
            drawdowns[column],
            color=color,
            linestyle=line_style,
            linewidth=line_width,
            label=label,
        )

    for index, column in enumerate(weights.columns):
        color = (
            weight_colors[column]
            if weight_colors is not None and column in weight_colors
            else BRAND_COLORS[index % len(BRAND_COLORS)]
        )
        weights_ax.plot(
            weights.index,
            weights[column],
            color=color,
            linewidth=2.2,
            label=column,
        )

    wealth_ax.set_title(title or "")
    wealth_ax.set_ylabel("Cumulative wealth")
    drawdown_ax.set_ylabel("Drawdown")
    weights_ax.set_ylabel("Target weight")
    drawdown_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    weights_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
        axis.set_xlim(returns.index.min(), returns.index.max())
    configure_time_axis(weights_ax)
    merge_legends(wealth_ax, ncol=min(3, len(returns.columns)))
    merge_legends(weights_ax, ncol=min(3, len(weights.columns)))
    if source:
        add_source_note(figure, source)
    _finish_figure(figure, source is not None)
    return figure, axes


def plot_underperformance_diagnostics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    rolling_window: int = 12,
    title: str | None = None,
    labels: tuple[str, str] | None = None,
    source: str | None = None,
    figsize: tuple[float, float] = (12.0, 8.0),
) -> tuple[Figure, tuple[Axes, Axes, Axes], pd.DataFrame]:
    """Show when a strategy loses value relative to its benchmark."""
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")
    strategy = _prepare_time_series(strategy_returns, "strategy_returns")
    benchmark = _prepare_time_series(benchmark_returns, "benchmark_returns")
    if not strategy.index.equals(benchmark.index):
        raise ValueError("strategy_returns and benchmark_returns need identical dates")
    if strategy.le(-1.0).any() or benchmark.le(-1.0).any():
        raise ValueError("returns cannot be less than or equal to -100%")
    if len(strategy) < rolling_window:
        raise ValueError("rolling_window is longer than the return sample")

    strategy_wealth = (1.0 + strategy).cumprod()
    benchmark_wealth = (1.0 + benchmark).cumprod()
    relative_wealth = strategy_wealth / benchmark_wealth
    relative_return = relative_wealth - 1.0
    relative_drawdown = _drawdown_from_wealth(relative_wealth)
    rolling_strategy = (1.0 + strategy).rolling(rolling_window).apply(np.prod, raw=True)
    rolling_benchmark = (1.0 + benchmark).rolling(rolling_window).apply(
        np.prod,
        raw=True,
    )
    rolling_relative_return = rolling_strategy / rolling_benchmark - 1.0
    diagnostics = pd.DataFrame(
        {
            "relative_return": relative_return,
            "relative_drawdown": relative_drawdown,
            f"rolling_{rolling_window}_period_relative_return": (
                rolling_relative_return
            ),
        }
    )

    strategy_label = (
        labels[0]
        if labels is not None
        else str(strategy_returns.name or "Strategy")
    )
    benchmark_label = (
        labels[1]
        if labels is not None
        else str(benchmark_returns.name or "Benchmark")
    )
    figure, axes_array = plt.subplots(3, 1, figsize=figsize, sharex=True)
    relative_ax, drawdown_ax, rolling_ax = axes_array
    axes = (relative_ax, drawdown_ax, rolling_ax)

    relative_ax.plot(
        diagnostics.index,
        diagnostics["relative_return"],
        color=BRAND_COLORS[0],
        linewidth=2.8,
    )
    drawdown_ax.plot(
        diagnostics.index,
        diagnostics["relative_drawdown"],
        color="#C83E4D",
        linewidth=2.2,
    )
    drawdown_ax.fill_between(
        diagnostics.index,
        diagnostics["relative_drawdown"],
        0.0,
        color="#F4CCCC",
        alpha=0.65,
    )
    rolling_column = f"rolling_{rolling_window}_period_relative_return"
    rolling_values = diagnostics[rolling_column]
    rolling_ax.plot(
        diagnostics.index,
        rolling_values,
        color=BRAND_COLORS[0],
        linewidth=2.2,
    )
    rolling_ax.fill_between(
        diagnostics.index,
        rolling_values,
        0.0,
        where=rolling_values.lt(0.0),
        color="#F4CCCC",
        alpha=0.75,
        interpolate=True,
    )

    relative_ax.set_title(title or f"{strategy_label} vs {benchmark_label}")
    relative_ax.set_ylabel("Cumulative relative return")
    drawdown_ax.set_ylabel("Relative drawdown")
    rolling_ax.set_ylabel(f"{rolling_window}-period relative return")
    relative_ax.axhline(0.0, color="#7A8793", linewidth=0.9)
    rolling_ax.axhline(0.0, color="#7A8793", linewidth=0.9)
    for axis in axes:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", alpha=0.18)
        axis.set_xlim(diagnostics.index.min(), diagnostics.index.max())
    configure_time_axis(rolling_ax)
    if source:
        add_source_note(figure, source)
    _finish_figure(figure, source is not None)
    return figure, axes, diagnostics


def plot_signal_confirmation(
    baseline_score: pd.Series,
    challenger_score: pd.Series,
    outcome_relative_series: pd.Series,
    *,
    disagreement_flags: pd.Series | None = None,
    labels: tuple[str, str, str] | None = None,
    outcome_reference: float = 1.0,
    title: str | None = None,
    source: str | None = None,
    figsize: tuple[float, float] = (12.0, 7.0),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Compare two prepared signals with an investable outcome series."""
    baseline = _prepare_time_series(baseline_score, "baseline_score")
    challenger = _prepare_time_series(challenger_score, "challenger_score")
    outcome = _prepare_time_series(
        outcome_relative_series,
        "outcome_relative_series",
    )
    if not baseline.index.equals(challenger.index) or not baseline.index.equals(
        outcome.index
    ):
        raise ValueError("all signal-confirmation inputs need identical dates")

    if disagreement_flags is None:
        disagreement = pd.Series(False, index=baseline.index)
    else:
        disagreement = _prepare_boolean_series(
            disagreement_flags,
            index=baseline.index,
            argument_name="disagreement_flags",
        )
    if labels is None:
        labels = (
            str(baseline_score.name or "Baseline"),
            str(challenger_score.name or "Challenger"),
            str(outcome_relative_series.name or "Relative outcome"),
        )

    figure, axes_array = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": (1.2, 1.0)},
    )
    signal_ax, outcome_ax = axes_array
    axes = (signal_ax, outcome_ax)
    signal_ax.plot(
        baseline.index,
        baseline,
        color=BRAND_COLORS[0],
        linewidth=2.6,
        label=labels[0],
    )
    signal_ax.plot(
        challenger.index,
        challenger,
        color=BRAND_COLORS[1],
        linewidth=2.3,
        label=labels[1],
    )
    outcome_ax.plot(
        outcome.index,
        outcome,
        color=BRAND_COLORS[2],
        linewidth=2.6,
        label=labels[2],
    )
    for timestamp in disagreement.loc[disagreement].index:
        start = timestamp.to_period("M").start_time
        end = timestamp.to_period("M").end_time
        for axis in axes:
            axis.axvspan(
                start,
                end,
                color="#BFBFBF",
                alpha=0.28,
                linewidth=0,
                zorder=0,
            )
    signal_ax.axhline(0.0, color="#7A8793", linewidth=0.9)
    outcome_ax.axhline(outcome_reference, color="#7A8793", linewidth=0.9)
    signal_ax.set_title(title or "")
    signal_ax.set_ylabel("Prepared signal / score")
    outcome_ax.set_ylabel(labels[2])
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
        axis.set_xlim(baseline.index.min(), baseline.index.max())
    configure_time_axis(outcome_ax)
    merge_legends(signal_ax, ncol=2)
    merge_legends(outcome_ax)
    if source:
        add_source_note(figure, source)
    _finish_figure(figure, source is not None)
    return figure, axes


def plot_regime_heatmap(
    regimes: pd.DataFrame,
    *,
    state_colors: Mapping[str, str],
    flags: pd.DataFrame | None = None,
    marker_flags: pd.DataFrame | None = None,
    outline_flags: pd.DataFrame | None = None,
    title: str | None = None,
    flag_symbol: str = "†",
    marker_symbol: str = "×",
    outline_color: str = "#343A40",
    missing_label: str = "N/A",
    highlight_latest: bool = True,
    source: str | None = None,
    figsize: tuple[float, float] = (14.0, 6.0),
) -> tuple[Figure, Axes]:
    """Plot a month-by-model categorical regime heatmap."""
    if not isinstance(regimes, pd.DataFrame):
        raise TypeError("regimes must be a pandas DataFrame")
    if not isinstance(regimes.index, pd.DatetimeIndex):
        raise TypeError("regimes.index must be a DatetimeIndex")
    if regimes.empty:
        raise ValueError("regimes cannot be empty")
    if regimes.index.has_duplicates or regimes.columns.has_duplicates:
        raise ValueError("regimes cannot contain duplicate index or columns")
    if not all(isinstance(column, str) for column in regimes.columns):
        raise TypeError("regimes.columns must contain strings")
    if not state_colors:
        raise ValueError("state_colors cannot be empty")

    states = regimes.sort_index().copy()
    states = states.map(lambda value: missing_label if pd.isna(value) else str(value))
    used_states = set(states.stack().unique())
    missing_colors = sorted(used_states.difference(state_colors))
    if missing_colors:
        raise ValueError(f"Missing colors for states: {missing_colors}")
    aligned_flags = _prepare_flag_frame(
        flags,
        index=states.index,
        columns=states.columns,
    )
    aligned_markers = _prepare_flag_frame(
        marker_flags,
        index=states.index,
        columns=states.columns,
    )
    aligned_outlines = _prepare_flag_frame(
        outline_flags,
        index=states.index,
        columns=states.columns,
    )

    state_order = list(state_colors)
    numeric_map = {state: index for index, state in enumerate(state_order)}
    matrix = states.T.map(numeric_map.__getitem__).to_numpy(dtype=float)
    color_map = ListedColormap([state_colors[state] for state in state_order])

    figure, axis = plt.subplots(figsize=figsize)
    axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        vmin=-0.5,
        vmax=len(state_order) - 0.5,
    )
    axis.set_yticks(range(len(states.columns)), labels=states.columns)
    axis.set_xticks(range(len(states.index)))
    axis.set_xticklabels(states.index.strftime("%Y-%m"))
    axis.tick_params(length=0)

    for row, model in enumerate(states.columns):
        for column, month in enumerate(states.index):
            state = states.loc[month, model]
            suffix = flag_symbol if aligned_flags.loc[month, model] else ""
            marker = marker_symbol if aligned_markers.loc[month, model] else ""
            axis.text(
                column,
                row,
                f"{state}{suffix}{marker}",
                ha="center",
                va="center",
                fontweight="bold",
                color="#343A40",
            )
            if aligned_outlines.loc[month, model]:
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor=outline_color,
                        linewidth=2.0,
                    )
                )
    for boundary in np.arange(0.5, len(states.columns), 1.0):
        axis.axhline(boundary, color="white", linewidth=1.5)
    if highlight_latest:
        axis.add_patch(
            Rectangle(
                (len(states.index) - 1.5, -0.5),
                1.0,
                len(states.columns),
                fill=False,
                edgecolor=BRAND_COLORS[0],
                linewidth=2.2,
            )
        )

    axis.set_title(title or "")
    axis.legend(
        handles=[
            Patch(facecolor=state_colors[state], label=state)
            for state in state_order
            if state in used_states
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(5, len(used_states)),
    )
    if source:
        add_source_note(figure, source)
    bottom = 0.16 if source is None else 0.20
    figure.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    return figure, axis


def plot_weighted_contribution_heatmap(
    component_scores: pd.DataFrame,
    component_weights: Mapping[str, float],
    *,
    flags: pd.DataFrame | None = None,
    display_names: Mapping[str, str] | None = None,
    title: str | None = None,
    total_label: str = "Composite score",
    flag_symbol: str = "†",
    precision: int = 2,
    negative_color: str = "#C83E4D",
    positive_color: str = "#5B9E45",
    highlight_latest: bool = True,
    source: str | None = None,
    figsize: tuple[float, float] = (14.0, 7.0),
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot component score times effective weight with a zero-centered scale."""
    scores = _prepare_numeric_frame(
        component_scores,
        "component_scores",
        require_complete=False,
    )
    supplied = set(component_weights)
    required = set(scores.columns)
    if supplied != required:
        missing = sorted(required.difference(supplied))
        extra = sorted(supplied.difference(required))
        raise ValueError(f"component_weights mismatch; missing={missing}, extra={extra}")
    weights = pd.Series(component_weights, dtype=float).reindex(scores.columns)
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("component_weights must be finite")
    if precision < 0:
        raise ValueError("precision cannot be negative")
    if total_label in scores.columns:
        raise ValueError("total_label must not match a component column")

    contributions = scores.mul(weights, axis=1)
    component_flags = _prepare_flag_frame(
        flags,
        index=scores.index,
        columns=scores.columns,
    )
    plot_data = contributions.copy()
    plot_data[total_label] = contributions.sum(
        axis=1,
        min_count=len(contributions.columns),
    )
    plot_flags = component_flags.copy()
    plot_flags[total_label] = component_flags.any(axis=1)

    row_keys = list(plot_data.columns)
    row_labels = [
        total_label
        if key == total_label
        else _display_name(key, display_names)
        for key in row_keys
    ]
    if len(row_labels) != len(set(row_labels)):
        raise ValueError("display_names must produce unique row labels")

    matrix = plot_data.T.to_numpy(dtype=float)
    finite_values = np.abs(matrix[np.isfinite(matrix)])
    limit = float(finite_values.max()) if finite_values.size else 1.0
    if limit == 0.0:
        limit = 1.0
    color_map = LinearSegmentedColormap.from_list(
        "weighted_contribution",
        [negative_color, "#F4CCCC", "#FFFFFF", "#D9EAD3", positive_color],
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        norm=norm,
    )
    axis.set_yticks(range(len(row_labels)), labels=row_labels)
    axis.set_xticks(range(len(plot_data.index)))
    axis.set_xticklabels(plot_data.index.strftime("%Y-%m"))
    axis.tick_params(length=0)

    for row, key in enumerate(row_keys):
        for column, month in enumerate(plot_data.index):
            value = plot_data.loc[month, key]
            if pd.isna(value):
                annotation = "N/A"
            else:
                suffix = flag_symbol if plot_flags.loc[month, key] else ""
                annotation = f"{value:+.{precision}f}{suffix}"
            axis.text(
                column,
                row,
                annotation,
                ha="center",
                va="center",
                fontweight="bold" if key == total_label else "normal",
                color="#343A40",
            )
    axis.axhline(len(row_keys) - 1.5, color="white", linewidth=2.5)
    if highlight_latest:
        axis.add_patch(
            Rectangle(
                (len(plot_data.index) - 1.5, -0.5),
                1.0,
                len(row_keys),
                fill=False,
                edgecolor=BRAND_COLORS[0],
                linewidth=2.2,
            )
        )

    axis.set_title(title or "")
    color_bar = figure.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        fraction=0.055,
        pad=0.13,
        aspect=45,
    )
    color_bar.set_label("Weighted contribution to composite score")
    if source:
        add_source_note(figure, source)
    _finish_figure(figure, source is not None)
    return figure, axis, plot_data


def save_figure(
    fig: Figure,
    output_path: str | Path,
    *,
    dpi: int = 300,
    create_parent: bool = True,
) -> Path:
    """Save a figure explicitly and return the resolved output path."""
    path = Path(output_path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".png")
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path.resolve()
