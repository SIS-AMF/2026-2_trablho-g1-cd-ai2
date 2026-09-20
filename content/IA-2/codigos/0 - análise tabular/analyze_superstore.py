#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = SCRIPT_DIR / ".mpl-cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


DATE_COL = "Data do Pedido"
STATE_COL = "Estado"
CATEGORY_COL = "Categoria"
SALES_COL = "Vendas"
DISCOUNT_COL = "Desconto"
PROFIT_COL = "Lucro"
ROLLING_WINDOW = 7
FORECAST_DAYS = 7


@dataclass
class ForecastResult:
    future_dates: pd.DatetimeIndex
    predicted_values: np.ndarray

    @property
    def total(self) -> float:
        return float(np.sum(self.predicted_values))


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%m/%d/%Y")
    df[SALES_COL] = pd.to_numeric(df[SALES_COL], errors="coerce").fillna(0.0)
    df[DISCOUNT_COL] = pd.to_numeric(df[DISCOUNT_COL], errors="coerce").fillna(0.0)
    df[PROFIT_COL] = pd.to_numeric(df[PROFIT_COL], errors="coerce").fillna(0.0)
    return df


def build_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(DATE_COL)
        .agg(
            {
                SALES_COL: "sum",
                PROFIT_COL: "sum",
                DISCOUNT_COL: "mean",
            }
        )
        .sort_index()
    )
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = DATE_COL
    return daily


def rolling_series(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def linear_forecast(series: pd.Series, forecast_days: int = FORECAST_DAYS, clip_lower: float | None = None) -> ForecastResult:
    smoothed = rolling_series(series)
    x_train = np.arange(len(smoothed)).reshape(-1, 1)
    y_train = smoothed.to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(x_train, y_train)

    x_future = np.arange(len(smoothed), len(smoothed) + forecast_days).reshape(-1, 1)
    predictions = model.predict(x_future)
    if clip_lower is not None:
        predictions = np.clip(predictions, clip_lower, None)

    future_dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    return ForecastResult(future_dates=future_dates, predicted_values=predictions)


def build_group_daily_series(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, pd.Series]:
    base_index = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    grouped = (
        df.groupby([DATE_COL, group_col])[value_col]
        .sum()
        .reset_index()
    )

    series_map: dict[str, pd.Series] = {}
    for group_name, group_df in grouped.groupby(group_col):
        series = group_df.set_index(DATE_COL)[value_col].sort_index()
        series = series.reindex(base_index, fill_value=0.0)
        series.index.name = DATE_COL
        series_map[str(group_name)] = series
    return series_map


def best_group_forecast(df: pd.DataFrame, group_col: str, value_col: str) -> tuple[str, ForecastResult]:
    forecasts: list[tuple[str, ForecastResult]] = []
    for group_name, series in build_group_daily_series(df, group_col, value_col).items():
        forecasts.append((group_name, linear_forecast(series, clip_lower=0.0)))
    return max(forecasts, key=lambda item: item[1].total)


def build_discount_profit_analysis(daily: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "desconto_mm7": rolling_series(daily[DISCOUNT_COL]),
            "lucro_mm7": rolling_series(daily[PROFIT_COL]),
        },
        index=daily.index,
    )


def discount_profit_effect(analysis: pd.DataFrame) -> tuple[str, float, LinearRegression]:
    model = LinearRegression()
    model.fit(analysis[["desconto_mm7"]], analysis["lucro_mm7"])
    coef = float(model.coef_[0])
    direction = "aumentar" if coef > 0 else "reduzir"
    return direction, coef, model


def build_discount_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(DISCOUNT_COL)[PROFIT_COL]
        .agg(["count", "mean", "sum"])
        .reset_index()
        .rename(columns={"count": "qtd_pedidos", "mean": "lucro_medio", "sum": "lucro_total"})
        .sort_values(DISCOUNT_COL)
    )
    return summary


def best_discount_level(summary: pd.DataFrame) -> tuple[float, float]:
    best_row = summary.loc[summary["lucro_medio"].idxmax()]
    return float(best_row[DISCOUNT_COL]), float(best_row["lucro_medio"])


def save_discount_profit_plot(summary: pd.DataFrame, output_path: Path) -> None:
    x = summary[DISCOUNT_COL].to_numpy(dtype=float)
    y = summary["lucro_medio"].to_numpy(dtype=float)
    counts = summary["qtd_pedidos"].to_numpy(dtype=float)
    best_discount, best_profit = best_discount_level(summary)

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    bars = ax1.bar(x, y, width=0.035, color="#4c78a8", alpha=0.85, label="Lucro medio por nivel de desconto")
    ax1.axhline(0, color="#444444", linewidth=1)
    ax1.scatter([best_discount], [best_profit], color="#d62728", s=90, zorder=5, label="Melhor ponto medio")
    ax1.annotate(
        f"Melhor media: {best_discount:.0%}\nLucro medio {format_currency(best_profit)}",
        xy=(best_discount, best_profit),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    ax1.set_title("Lucro medio por nivel de desconto")
    ax1.set_xlabel("Nivel de desconto")
    ax1.set_ylabel("Lucro medio por registro")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{value:.0%}" for value in x])
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(x, counts, color="#f28e2b", marker="o", linewidth=2, label="Quantidade de registros")
    ax2.set_ylabel("Quantidade de registros")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analise preditiva do Sample Superstore localizado para PT-BR.")
    parser.add_argument(
        "--input",
        default="VendasXPTO.csv",
        help="Caminho do CSV de entrada.",
    )
    parser.add_argument(
        "--plot-output",
        default="desconto_vs_lucro.png",
        help="Arquivo PNG para o grafico de desconto x lucro.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = SCRIPT_DIR / input_path

    plot_path = Path(args.plot_output)
    if not plot_path.is_absolute():
        plot_path = SCRIPT_DIR / plot_path

    df = load_data(str(input_path))
    daily = build_daily_frame(df)

    profit_forecast = linear_forecast(daily[PROFIT_COL])
    best_state, state_forecast = best_group_forecast(df, STATE_COL, SALES_COL)
    best_category, category_forecast = best_group_forecast(df, CATEGORY_COL, SALES_COL)
    discount_analysis = build_discount_profit_analysis(daily)
    discount_direction, discount_coef, discount_model = discount_profit_effect(discount_analysis)
    discount_summary = build_discount_level_summary(df)
    best_discount, best_discount_profit = best_discount_level(discount_summary)
    save_discount_profit_plot(discount_summary, plot_path)

    print("1. Nos proximos 7 dias, qual a previsao de lucro da empresa com base no historico?")
    for date, value in zip(profit_forecast.future_dates, profit_forecast.predicted_values):
        print(f"   {date.strftime('%d/%m/%Y')}: {format_currency(value)}")
    print(f"   Lucro total previsto em 7 dias: {format_currency(profit_forecast.total)}")

    print("\n2. Nos proximos 7 dias, qual o estado deve gerar mais faturamento e qual faturamento previsto? Qual categoria e qual faturamento previsto?")
    print(f"   Estado com maior faturamento previsto: {best_state} ({format_currency(state_forecast.total)})")
    print(f"   Categoria com maior faturamento previsto: {best_category} ({format_currency(category_forecast.total)})")

    print("\n3. Aumentar os descontos tende a aumentar ou reduzir o lucro?")
    print(f"   Pelos dados diarios com media movel de 7 dias, aumentar os descontos tende a {discount_direction} o lucro.")
    print(f"   Observando os niveis de desconto praticados, o melhor lucro medio aparece em {best_discount:.0%} de desconto, com {format_currency(best_discount_profit)} por registro.")
    print(f"   Acima de {best_discount:.0%}, o lucro medio passa a piorar no historico observado.")
    print(f"   Coeficiente da regressao linear (lucro x desconto): {discount_coef:.2f}")
    print(f"   Grafico salvo em: {plot_path}")


if __name__ == "__main__":
    main()
