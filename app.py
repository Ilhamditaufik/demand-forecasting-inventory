
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output


# =========================================================
# 1. KONFIGURASI DAN MEMBACA HASIL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"

required_files = {
    "summary": REPORT_DIR / "inventory_optimization_summary.csv",
    "stores": REPORT_DIR / "inventory_summary_by_store.csv",
    "recommendations": REPORT_DIR / "inventory_optimization_recommendations.csv",
    "models": REPORT_DIR / "national_aggregate_model_comparison.csv",
    "forecast": REPORT_DIR / "prophet_daily_forecast.csv",
}

for name, path in required_files.items():
    if not path.exists():
        raise FileNotFoundError(
            f"File {name} tidak ditemukan: {path}\n"
            "Jalankan tahap analisis dan optimasi terlebih dahulu."
        )

summary_df = pd.read_csv(required_files["summary"], dtype=str)
store_df = pd.read_csv(required_files["stores"])
recommendations_df = pd.read_csv(required_files["recommendations"])
models_df = pd.read_csv(required_files["models"])
forecast_df = pd.read_csv(
    required_files["forecast"],
    parse_dates=["date"],
)

summary = dict(zip(summary_df["metric"], summary_df["value"]))


def get_number(key, default=0.0):
    try:
        return float(summary.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def fmt_int(value):
    return f"{value:,.0f}"


def fmt_money(value):
    return f"${value:,.2f}"


def make_table(frame, columns, labels=None, max_rows=15):
    """Membuat tabel HTML tanpa komponen DataTable yang deprecated."""
    labels = labels or {}
    shown = frame[columns].head(max_rows)

    header = html.Thead(
        html.Tr([
            html.Th(
                labels.get(col, col.replace("_", " ").title()),
                style={
                    "padding": "12px",
                    "textAlign": "left",
                    "borderBottom": "1px solid #334155",
                    "whiteSpace": "nowrap",
                },
            )
            for col in columns
        ])
    )

    rows = []
    for _, row in shown.iterrows():
        cells = []
        for col in columns:
            value = row[col]

            if pd.isna(value):
                value = "—"
            elif isinstance(value, float):
                value = f"{value:,.2f}"

            cells.append(
                html.Td(
                    str(value),
                    style={
                        "padding": "10px 12px",
                        "borderBottom": "1px solid #243247",
                        "whiteSpace": "nowrap",
                    },
                )
            )

        rows.append(html.Tr(cells))

    return html.Div(
        html.Table(
            [header, html.Tbody(rows)],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "fontSize": "13px",
            },
        ),
        style={"overflowX": "auto"},
    )


# =========================================================
# 2. GAYA DASHBOARD
# =========================================================

BG = "#0b1220"
PANEL = "#121c2d"
TEXT = "#e5edf8"
MUTED = "#9aabc2"
BORDER = "#28364b"
ACCENT = "#60a5fa"

CARD_STYLE = {
    "background": PANEL,
    "padding": "20px",
    "border": f"1px solid {BORDER}",
    "borderRadius": "14px",
    "minWidth": "190px",
    "flex": "1",
}

PANEL_STYLE = {
    "background": PANEL,
    "padding": "18px",
    "border": f"1px solid {BORDER}",
    "borderRadius": "14px",
    "minWidth": "0",
}

FIGURE_STYLE = {
    "height": "350px",
    "backgroundColor": PANEL,
}

FIGURE_CONFIG = {
    "displaylogo": False,
    "responsive": True,
}


def kpi_card(title, value, subtitle):
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "color": MUTED,
                    "fontSize": "13px",
                    "marginBottom": "12px",
                },
            ),
            html.Div(
                value,
                style={
                    "color": TEXT,
                    "fontSize": "27px",
                    "fontWeight": "700",
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                subtitle,
                style={"color": MUTED, "fontSize": "12px"},
            ),
        ],
        style=CARD_STYLE,
    )


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": MUTED},
    )
    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": TEXT},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    return fig


# =========================================================
# 3. GRAFIK EVALUASI MODEL
# =========================================================

model_fig = px.bar(
    models_df.sort_values("WAPE_percent"),
    x="model",
    y="WAPE_percent",
    text="WAPE_percent",
    title="Model Forecasting — WAPE",
    labels={
        "model": "Model",
        "WAPE_percent": "WAPE (%)",
    },
)

model_fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
)

model_fig.update_layout(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font={"color": TEXT},
    title_font={"size": 16},
    margin={"l": 20, "r": 20, "t": 55, "b": 20},
    yaxis={"rangemode": "tozero", "gridcolor": BORDER},
    xaxis={"title": ""},
)

# Grafik penjualan aktual, prediksi, dan interval Prophet.
forecast_fig = go.Figure()

if {"yhat_lower", "yhat_upper"}.issubset(forecast_df.columns):
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["yhat_upper"].clip(lower=0),
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
            name="Batas atas interval",
        )
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["yhat_lower"].clip(lower=0),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(96,165,250,0.15)",
            name="Interval prediksi Prophet",
        )
    )

forecast_series = [
    ("actual_units", "Aktual", "solid"),
    ("seasonal_naive_forecast", "Seasonal Naive", "dot"),
    ("lightgbm_forecast", "LightGBM", "dash"),
    ("prophet_forecast", "Prophet", "solid"),
]

for column, label, dash_style in forecast_series:
    if column in forecast_df.columns:
        forecast_fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df[column],
                mode="lines",
                name=label,
                line={"width": 2, "dash": dash_style},
            )
        )

forecast_fig.update_layout(
    title="Backtest Nasional — 28 Hari",
    xaxis_title="Tanggal",
    yaxis_title="Total unit penjualan harian",
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font={"color": TEXT},
    margin={"l": 25, "r": 20, "t": 55, "b": 30},
    legend={"orientation": "h", "y": -0.2},
    xaxis={"gridcolor": BORDER},
    yaxis={"gridcolor": BORDER},
)


# =========================================================
# 4. DASH APP DAN LAYOUT
# =========================================================

app = Dash(__name__)
app.title = "Demand Forecasting & Inventory Optimization"

states = sorted(
    recommendations_df["state_id"].dropna().astype(str).unique()
)

state_options = [
    {"label": "Semua State", "value": "ALL"}
] + [
    {"label": state, "value": state}
    for state in states
]

app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.Div(
                    "DATA SCIENCE  /  CAPSTONE PROJECT DS-02",
                    style={
                        "color": ACCENT,
                        "fontSize": "11px",
                        "letterSpacing": "2px",
                        "fontWeight": "700",
                    },
                ),
                html.H1(
                    "Demand Forecasting & Inventory Optimization",
                    style={
                        "color": TEXT,
                        "fontSize": "30px",
                        "margin": "12px 0 8px",
                    },
                ),
                html.P(
                    "M5 Walmart dataset · LightGBM · Prophet · OR-Tools",
                    style={"color": MUTED, "margin": "0"},
                ),
            ],
            style={"marginBottom": "26px"},
        ),

        # KPIs
        html.Div(
            [
                kpi_card(
                    "Permintaan Prediksi (7 hari)",
                    fmt_int(get_number("forecast_demand_units")),
                    "Agregat 30.490 seri produk–toko",
                ),
                kpi_card(
                    "Rekomendasi Pemesanan",
                    fmt_int(get_number("recommended_order_units")),
                    "Unit hasil optimasi OR-Tools",
                ),
                kpi_card(
                    "Potensi Lost Sales",
                    fmt_int(get_number("potential_lost_sales_units")),
                    f"{get_number('potential_lost_sales_rate_pct'):.2f}% dari permintaan",
                ),
                kpi_card(
                    "Estimasi Biaya Pembelian",
                    fmt_money(get_number("estimated_purchase_cost")),
                    "Estimasi simulasi, bukan biaya aktual",
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "14px",
                "marginBottom": "22px",
            },
        ),

        # Charts
        html.Div(
            [
                html.Div(
                    [dcc.Graph(
                        id="forecast-chart",
                        figure=forecast_fig,
                        config=FIGURE_CONFIG,
                        style=FIGURE_STYLE,
                    )],
                    style={**PANEL_STYLE, "flex": "2"},
                ),
                html.Div(
                    [dcc.Graph(
                        id="model-chart",
                        figure=model_fig,
                        config=FIGURE_CONFIG,
                        style=FIGURE_STYLE,
                    )],
                    style={**PANEL_STYLE, "flex": "1"},
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "16px",
                "marginBottom": "22px",
            },
        ),

        # Filters
        html.Div(
            [
                html.H3(
                    "Inventory Explorer",
                    style={"color": TEXT, "marginTop": "0"},
                ),
                html.P(
                    "Pilih wilayah untuk melihat rekomendasi berdasarkan "
                    "prediksi tujuh hari dan skenario stok.",
                    style={"color": MUTED, "fontSize": "13px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "State",
                                    style={
                                        "color": MUTED,
                                        "display": "block",
                                        "marginBottom": "8px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="state-filter",
                                    options=state_options,
                                    value="ALL",
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "180px"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Toko",
                                    style={
                                        "color": MUTED,
                                        "display": "block",
                                        "marginBottom": "8px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="store-filter",
                                    options=[
                                        {"label": "Semua Toko", "value": "ALL"}
                                    ],
                                    value="ALL",
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "180px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "18px",
                    },
                ),
            ],
            style={**PANEL_STYLE, "marginBottom": "22px"},
        ),

        # Inventory charts
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="store-chart",
                            figure=empty_figure("Memuat ringkasan toko..."),
                            config=FIGURE_CONFIG,
                            style=FIGURE_STYLE,
                        )
                    ],
                    style={**PANEL_STYLE, "flex": "1"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="recommendation-chart",
                            figure=empty_figure("Memuat rekomendasi..."),
                            config=FIGURE_CONFIG,
                            style=FIGURE_STYLE,
                        )
                    ],
                    style={**PANEL_STYLE, "flex": "1"},
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "16px",
                "marginBottom": "22px",
            },
        ),

        # Tables
        html.Div(
            [
                html.H3(
                    "Ringkasan per Toko",
                    style={"color": TEXT, "marginTop": "0"},
                ),
                html.Div(id="store-table"),
            ],
            style={**PANEL_STYLE, "marginBottom": "22px"},
        ),

        html.Div(
            [
                html.H3(
                    "Rekomendasi Produk–Toko",
                    style={"color": TEXT, "marginTop": "0"},
                ),
                html.P(
                    "Tabel diurutkan berdasarkan permintaan prediksi tertinggi.",
                    style={"color": MUTED, "fontSize": "12px"},
                ),
                html.Div(id="recommendation-table"),
            ],
            style={**PANEL_STYLE, "marginBottom": "22px"},
        ),

        # Disclosure and footer
        html.Div(
            [
                html.H4("Catatan dan Batasan", style={"color": TEXT}),
                html.P(
                    "Dashboard ini menggunakan data historis dan asumsi simulasi. "
                    "Stok awal, biaya pembelian, biaya penyimpanan, penalti "
                    "kehabisan stok, anggaran, dan kapasitas bukan data operasional "
                    "aktual Walmart. Backtest nasional dan simulasi pesanan "
                    "merupakan dua hasil evaluasi yang berbeda.",
                    style={
                        "color": MUTED,
                        "fontSize": "13px",
                        "lineHeight": "1.7",
                    },
                ),
                html.P(
                    "Capstone DS-02 · Demand Forecasting & Inventory Optimization",
                    style={
                        "color": ACCENT,
                        "fontSize": "12px",
                        "marginTop": "20px",
                    },
                ),
            ],
            style={**PANEL_STYLE, "marginBottom": "20px"},
        ),
    ],
    style={
        "background": BG,
        "minHeight": "100vh",
        "padding": "28px",
        "fontFamily": "Arial, sans-serif",
        "color": TEXT,
    },
)


# =========================================================
# 5. CALLBACK FILTER TOKO
# =========================================================

@app.callback(
    Output("store-filter", "options"),
    Output("store-filter", "value"),
    Input("state-filter", "value"),
)
def update_store_options(selected_state):
    filtered = store_df.copy()
    filtered["state_id"] = filtered["state_id"].astype(str)
    filtered["store_id"] = filtered["store_id"].astype(str)

    if selected_state != "ALL":
        filtered = filtered[
            filtered["state_id"] == selected_state
        ]

    stores_for_state = sorted(filtered["store_id"].unique())

    options = [
        {"label": "Semua Toko", "value": "ALL"}
    ] + [
        {"label": store, "value": store}
        for store in stores_for_state
    ]

    return options, "ALL"


# =========================================================
# 6. CALLBACK GRAFIK DAN TABEL INVENTARIS
# =========================================================

@app.callback(
    Output("store-chart", "figure"),
    Output("recommendation-chart", "figure"),
    Output("store-table", "children"),
    Output("recommendation-table", "children"),
    Input("state-filter", "value"),
    Input("store-filter", "value"),
)
def update_inventory_dashboard(selected_state, selected_store):
    stores = store_df.copy()
    recs = recommendations_df.copy()

    for frame in (stores, recs):
        frame["state_id"] = frame["state_id"].astype(str)
        frame["store_id"] = frame["store_id"].astype(str)

    if selected_state != "ALL":
        stores = stores[
            stores["state_id"] == selected_state
        ]
        recs = recs[
            recs["state_id"] == selected_state
        ]

    if selected_store != "ALL":
        stores = stores[
            stores["store_id"] == selected_store
        ]
        recs = recs[
            recs["store_id"] == selected_store
        ]

    # Grafik pesanan menurut toko.
    if stores.empty:
        store_fig = empty_figure("Tidak ada data untuk filter ini.")
    else:
        top_stores = (
            stores.sort_values(
                "recommended_order_units",
                ascending=False,
            )
            .head(12)
            .sort_values("recommended_order_units")
        )

        store_fig = px.bar(
            top_stores,
            x="recommended_order_units",
            y="store_id",
            color="state_id",
            orientation="h",
            title="Rekomendasi Pesanan per Toko",
            labels={
                "recommended_order_units": "Unit pesanan",
                "store_id": "Toko",
                "state_id": "State",
            },
        )

        store_fig.update_layout(
            paper_bgcolor=PANEL,
            plot_bgcolor=PANEL,
            font={"color": TEXT},
            margin={"l": 20, "r": 20, "t": 50, "b": 25},
            xaxis={"gridcolor": BORDER},
            yaxis={"gridcolor": BORDER},
        )

    # Grafik perbandingan permintaan dan pesanan.
    if recs.empty:
        rec_fig = empty_figure("Tidak ada rekomendasi untuk filter ini.")
    else:
        top_recs = (
            recs.sort_values(
                "planning_demand_7d",
                ascending=False,
            )
            .head(12)
            .copy()
        )

        top_recs["series_label"] = (
            top_recs["item_id"].astype(str)
            + " / "
            + top_recs["store_id"].astype(str)
        )

        melted = top_recs.melt(
            id_vars=["series_label"],
            value_vars=[
                "planning_demand_7d",
                "recommended_order_units",
            ],
            var_name="measure",
            value_name="units",
        )

        measure_labels = {
            "planning_demand_7d": "Permintaan",
            "recommended_order_units": "Pesanan",
        }
        melted["measure"] = melted["measure"].map(measure_labels)

        rec_fig = px.bar(
            melted,
            x="units",
            y="series_label",
            color="measure",
            barmode="group",
            orientation="h",
            title="Permintaan vs Rekomendasi Pesanan",
            labels={
                "units": "Unit",
                "series_label": "Produk / Toko",
                "measure": "Metrik",
            },
        )

        rec_fig.update_layout(
            paper_bgcolor=PANEL,
            plot_bgcolor=PANEL,
            font={"color": TEXT},
            margin={"l": 20, "r": 20, "t": 50, "b": 25},
            xaxis={"gridcolor": BORDER},
            yaxis={"gridcolor": BORDER},
            legend={"orientation": "h", "y": -0.2},
        )

    # Tabel ringkasan toko.
    store_table = make_table(
        stores.sort_values(
            "recommended_order_units",
            ascending=False,
        ),
        [
            "state_id",
            "store_id",
            "forecast_demand_units",
            "assumed_initial_stock",
            "recommended_order_units",
            "lost_sales_units",
            "ending_inventory_units",
            "estimated_purchase_cost",
        ],
        labels={
            "state_id": "State",
            "store_id": "Toko",
            "forecast_demand_units": "Permintaan",
            "assumed_initial_stock": "Stok Awal",
            "recommended_order_units": "Rekomendasi Pesanan",
            "lost_sales_units": "Potensi Lost Sales",
            "ending_inventory_units": "Stok Akhir",
            "estimated_purchase_cost": "Estimasi Biaya ($)",
        },
        max_rows=15,
    )

    # Tabel rekomendasi detail.
    recommendation_table = make_table(
        recs.sort_values(
            "planning_demand_7d",
            ascending=False,
        ),
        [
            "item_id",
            "store_id",
            "planning_demand_7d",
            "assumed_initial_stock",
            "recommended_order_units",
            "expected_lost_sales_units",
            "expected_ending_inventory_units",
        ],
        labels={
            "item_id": "Produk",
            "store_id": "Toko",
            "planning_demand_7d": "Permintaan 7 Hari",
            "assumed_initial_stock": "Stok Awal",
            "recommended_order_units": "Pesanan",
            "expected_lost_sales_units": "Potensi Lost Sales",
            "expected_ending_inventory_units": "Stok Akhir",
        },
        max_rows=25,
    )

    return store_fig, rec_fig, store_table, recommendation_table


# =========================================================
# 7. JALANKAN APLIKASI
# =========================================================

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
