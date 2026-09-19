"""Interview-ready, read-only Streamlit application over versioned pipeline artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import OUTPUTS  # noqa: E402

st.set_page_config(
    page_title="Enterprise CLV | Customer Intelligence", page_icon="◈", layout="wide"
)
st.markdown(
    """<style>
    .block-container {padding-top:4.2rem; padding-bottom:2rem; max-width:1500px;}
    h1 {letter-spacing:-1.3px; font-weight:750 !important;}
    h2,h3 {letter-spacing:-.4px;}
    [data-testid="stMetric"] {background:white; border:1px solid #E1E8EE; border-radius:12px; padding:14px; min-height:105px;}
    [data-testid="stMetricLabel"] {color:#607182; font-size:12px;}
    [data-testid="stMetricLabel"] p {white-space:normal; overflow:visible; text-overflow:unset;}
    [data-testid="stMetricValue"] {font-size:27px;}
    .eyebrow {color:#0F766E; font-size:12px; font-weight:750; letter-spacing:2px; margin-bottom:6px;}
    .subtitle {color:#607182; font-size:15px; margin-top:-8px; margin-bottom:22px;}
    [data-testid="stSidebar"] {border-right:1px solid #E1E8EE;}
</style>""",
    unsafe_allow_html=True,
)

COLORS = ["#0F766E", "#64748B", "#F59E0B", "#E76F51", "#6366F1"]
MODEL_COLORS = {"baseline": "#94A3B8", "cox_raw": "#6366F1", "cox_calibrated": "#0F766E"}
px.defaults.color_discrete_sequence = COLORS


def money(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:,.1f}K"
    return f"${value:,.0f}"


def plot(fig: go.Figure, height: int = 350) -> None:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#425466", size=12),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
        title_font=dict(size=16, color="#142536"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E7EDF2", zeroline=False)
    st.plotly_chart(fig, width="stretch")


@st.cache_data
def load_artifacts(signature: tuple) -> dict:
    _ = signature  # mtimes invalidate the cache after retraining/scoring
    names = ["dashboard_accounts", "survival_curves", "risk_explanations"]
    data = {name: pd.read_parquet(OUTPUTS / f"{name}.parquet") for name in names}
    for name in [
        "test_financial",
        "test_brier",
        "test_calibration",
        "validation_brier",
        "array_competing_risks",
        "kaplan_meier",
        "cox_coefficients",
    ]:
        data[name] = pd.read_csv(OUTPUTS / f"{name}.csv")
    data["metrics"] = json.loads((OUTPUTS / "metrics.json").read_text())
    data["manifest"] = json.loads((OUTPUTS / "scoring_manifest.json").read_text())
    return data


required = [
    OUTPUTS / f"{name}.parquet"
    for name in ["dashboard_accounts", "survival_curves", "risk_explanations"]
]
required += [
    OUTPUTS / f"{name}.csv"
    for name in [
        "test_financial",
        "test_brier",
        "test_calibration",
        "validation_brier",
        "array_competing_risks",
        "kaplan_meier",
        "cox_coefficients",
    ]
]
required += [OUTPUTS / "metrics.json", OUTPUTS / "scoring_manifest.json"]
if not all(path.exists() for path in required):
    st.title("Enterprise CLV & Customer Survival")
    st.info("Prepare the synthetic data and model artifacts to open the dashboard.")
    st.code(
        "python -m src.generate_data\npython -m src.train\npython -m src.score", language="bash"
    )
    st.stop()

data = load_artifacts(tuple(path.stat().st_mtime_ns for path in required))
all_accounts = data["dashboard_accounts"]
if data["manifest"]["model_version"] != data["metrics"]["model_version"]:
    st.error(
        "Scores and validation artifacts refer to different models. Run python -m src.score again."
    )
    st.stop()
st.sidebar.markdown("## ◈ Enterprise CLV")
st.sidebar.caption("CUSTOMER INTELLIGENCE PLATFORM")
page = st.sidebar.radio(
    "Workspace",
    ["Portfolio Overview", "Risk Explorer", "Customer Detail", "Model Validation"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown("**Portfolio filters**")
segment = st.sidebar.multiselect("Customer segment", sorted(all_accounts.company_size.unique()))
product = st.sidebar.multiselect("Product family", sorted(all_accounts.product_family.unique()))
region = st.sidebar.multiselect("Region", sorted(all_accounts.region.unique()))
threshold = st.sidebar.slider("High-risk threshold · 12m", 0.05, 0.6, 0.20, 0.05, format="%.2f")
accounts = all_accounts.copy()
for col, values in [("company_size", segment), ("product_family", product), ("region", region)]:
    if values:
        accounts = accounts[accounts[col].isin(values)]
st.sidebar.divider()
st.sidebar.caption(
    f"AS OF {data['manifest']['scoring_date']}  •  USD\n\nSynthetic data · {data['manifest']['backend'].upper()}\n\nModel: {data['manifest']['model_version']}"
)
st.markdown(
    '<div class="eyebrow">ENTERPRISE ANALYTICS / RECURRING REVENUE</div>', unsafe_allow_html=True
)
st.title(page)
subtitles = {
    "Portfolio Overview": "A forward view of customer retention, recurring contribution, and portfolio exposure.",
    "Risk Explorer": "Explore risk across the installed base and compare the timing of potential churn.",
    "Customer Detail": "Understand one account’s outlook and the model-supported drivers behind its risk.",
    "Model Validation": "A frozen temporal backtest: predictions at December 2024, observed outcomes through December 2025.",
}
st.markdown(f'<div class="subtitle">{subtitles[page]}</div>', unsafe_allow_html=True)
if accounts.empty and page != "Model Validation":
    st.info("No accounts match these filters. Clear one or more filters in the sidebar.")
    st.stop()

if page == "Portfolio Overview":
    cols = st.columns(5)
    for col, label, value in zip(
        cols,
        [
            "Active customers",
            "Current ARR",
            "CLV · 24m",
            "ARR at risk · 12m",
            "High-risk accounts",
        ],
        [
            f"{len(accounts):,}",
            money(accounts.ARR.sum()),
            money(accounts.predicted_clv.sum()),
            money(accounts.revenue_at_risk.sum()),
            f"{(accounts.churn_risk_12m >= threshold).sum():,}",
        ],
        strict=True,
    ):
        col.metric(label, value)
    st.caption(
        "CLV = discounted recurring contribution over 24 months. Revenue at risk = current ARR × 12-month churn probability; it is an exposure proxy, not a forecast of lost revenue."
    )
    left, right = st.columns([1.2, 1])
    with left:
        summary = accounts.groupby("company_size", as_index=False).agg(
            clv=("predicted_clv", "sum"), baseline=("baseline_clv", "sum")
        )
        chart = summary.melt("company_size", var_name="Model", value_name="Value")
        chart["Model"] = chart.Model.map(
            {"clv": "Survival-based CLV", "baseline": "Financial baseline"}
        )
        plot(
            px.bar(
                chart,
                x="company_size",
                y="Value",
                color="Model",
                barmode="group",
                title="24-month value by customer segment",
                labels={"company_size": "", "Value": "Discounted contribution · USD"},
            )
        )
    with right:
        summary = (
            accounts.groupby("product_family", as_index=False)
            .revenue_at_risk.sum()
            .sort_values("revenue_at_risk")
        )
        plot(
            px.bar(
                summary,
                y="product_family",
                x="revenue_at_risk",
                orientation="h",
                title="ARR exposure by product family",
                labels={"product_family": "", "revenue_at_risk": "Risk-weighted ARR · USD"},
            )
        )
    st.subheader("Accounts to review")
    st.caption(
        "Ranked by risk-weighted ARR. Use Customer Detail to inspect the survival curve and contribution drivers."
    )
    top = accounts.sort_values("revenue_at_risk", ascending=False).head(15)
    st.dataframe(
        top[
            [
                "customer_id",
                "company_size",
                "product_family",
                "ARR",
                "churn_risk_12m",
                "predicted_clv",
                "top_risk_driver",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "ARR": st.column_config.NumberColumn("ARR", format="$%,.0f"),
            "churn_risk_12m": st.column_config.ProgressColumn(
                "12m churn risk", min_value=0, max_value=1, format="percent"
            ),
            "predicted_clv": st.column_config.NumberColumn("24m CLV", format="$%,.0f"),
        },
    )
    st.download_button(
        "Download filtered portfolio", accounts.to_csv(index=False), "portfolio.csv", "text/csv"
    )

elif page == "Risk Explorer":
    left, right = st.columns(2)
    with left:
        plot(
            px.histogram(
                accounts,
                x="churn_risk_12m",
                color="company_size",
                nbins=30,
                barmode="stack",
                title="Distribution of 12-month churn risk",
                labels={"churn_risk_12m": "Churn probability", "company_size": "Segment"},
            )
        )
    with right:
        plot(
            px.scatter(
                accounts,
                x="churn_risk_12m",
                y="ARR",
                color="company_size",
                hover_name="customer_id",
                opacity=0.6,
                title="Risk and revenue concentration",
                labels={"churn_risk_12m": "12-month churn probability", "ARR": "Current ARR · USD"},
            )
        )
    left, right = st.columns(2)
    with left:
        curves = data["survival_curves"].merge(
            accounts[["customer_id", "company_size"]], on="customer_id"
        )
        fig = go.Figure()
        for i, (name, group) in enumerate(curves.groupby("company_size")):
            y = group[[f"month_{t}" for t in range(1, 25)]].mean().to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(25), y=np.r_[1, y], name=name, line=dict(color=COLORS[i], width=3)
                )
            )
        fig.update_layout(
            title="Mean predicted survival by segment",
            xaxis_title="Months after scoring",
            yaxis_title="Probability active",
        )
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
        plot(fig)
    with right:
        cif = data["array_competing_risks"]
        plot(
            px.line(
                cif,
                x="month",
                y=["customer_churn", "array_replacement", "end_of_service"],
                title="Array first-event cumulative incidence · all history",
                labels={"value": "Cumulative probability", "month": "Months since installation"},
            )
        )
        st.caption(
            "Array-level Aalen–Johansen estimates use the full synthetic history and ignore portfolio filters. Replacement ends an asset spell, not the customer relationship."
        )

elif page == "Customer Detail":
    chosen = st.selectbox(
        "Choose an account",
        accounts.sort_values("revenue_at_risk", ascending=False).customer_id.tolist(),
    )
    row = accounts.set_index("customer_id").loc[chosen]
    st.markdown(f"### {chosen}  ·  {row.company_size}")
    st.caption(
        f"{row.industry}  /  {row.region}  /  {row.product_family}  ·  Customer since {row.account_start_date:%b %Y}  ·  {int(row.number_of_arrays)} active arrays"
    )
    cols = st.columns(4)
    for col, h in zip(cols, [3, 6, 12, 24], strict=True):
        col.metric(f"{h}-month churn risk", f"{row[f'churn_risk_{h}m']:.1%}")
    value_cols = st.columns(2)
    value_cols[0].metric("Predicted CLV · 24m", money(row.predicted_clv))
    value_cols[1].metric("Current ARR", money(row.ARR))
    left, right = st.columns([1.1, 1])
    with left:
        curve = data["survival_curves"].set_index("customer_id").loc[chosen].to_numpy(dtype=float)
        fig = go.Figure(
            go.Scatter(
                x=np.arange(25),
                y=np.r_[1, curve],
                line=dict(color=COLORS[0], width=3, shape="hv"),
                fill="tozeroy",
                fillcolor="rgba(15,118,110,0.08)",
            )
        )
        fig.update_layout(
            title="Predicted customer survival",
            xaxis_title="Months after scoring",
            yaxis_title="Probability active",
        )
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
        plot(fig)
        st.caption(
            f"Restricted expected active months: {row.expected_remaining_lifetime:.1f} of 24. This is not an estimate of total remaining lifetime. 24m probabilities have only 12m out-of-time validation."
        )
    with right:
        terms = data["risk_explanations"].loc[lambda x: x.customer_id == chosen]
        selected = (
            pd.concat(
                [
                    terms.nlargest(5, "log_hazard_contribution"),
                    terms.nsmallest(5, "log_hazard_contribution"),
                ]
            )
            .drop_duplicates("feature")
            .sort_values("log_hazard_contribution")
        )
        driver_labels = {
            "storage_utilization": "Storage utilization",
            "ARR": "Current ARR",
            "previous_renewals": "Previous renewals",
            "account_age_months": "Account age",
            "support_tickets_30d": "Support tickets · 1m",
            "support_tickets_90d": "Support tickets · 3m",
            "critical_incidents_90d": "Critical incidents · 3m",
            "data_growth": "Usage growth",
            "hardware_age": "Hardware age",
            "company_size_Strategic": "Strategic segment",
            "company_size_Mid-market": "Mid-market segment",
            "months_until_end_of_service": "Time until EOS",
            "number_of_previous_expansions": "Previous expansions",
            "number_of_arrays": "Installed arrays",
            "total_capacity_tb": "Installed capacity",
            "utilization_3m_change": "Usage change · 3m",
            "utilization_6m_slope": "Usage trend · 6m",
            "ARR_growth_12m": "ARR growth · 12m",
            "expansion_last_12m": "Expansion · 12m",
            "months_until_renewal": "Time until renewal",
        }
        selected["driver"] = selected.feature.map(driver_labels).fillna(
            selected.feature.str.replace("industry_", "", regex=False)
            .str.replace("product_family_", "", regex=False)
            .str.replace("region_", "", regex=False)
        )
        plot(
            px.bar(
                selected,
                y="driver",
                x="log_hazard_contribution",
                color="direction",
                orientation="h",
                title="Model-supported risk drivers",
                color_discrete_map={
                    "Increases risk": "#E76F51",
                    "Decreases risk": "#0F766E",
                    "Neutral": "#94A3B8",
                },
                labels={
                    "driver": "",
                    "log_hazard_contribution": "Contribution to log relative hazard",
                },
            ),
            410,
        )
        st.caption(
            "Exact Cox coefficient × centered feature contributions relative to the training mean. These explain the model, not causal business effects. Scalar calibration preserves the relative-hazard ranking."
        )
    with st.expander("Inspect the feature snapshot"):
        st.dataframe(row.to_frame("Value").astype(str), width="stretch")

else:
    st.caption(
        "Validation results use the entire locked test cohort. Portfolio filters and risk threshold do not change these measurements."
    )
    metrics, brier = data["metrics"], data["test_brier"]
    brier12 = brier[brier.horizon_months == 12].set_index("model").brier_score
    cols = st.columns(4)
    cols[0].metric("Cox C-index · test", f"{metrics['test']['c_index']:.3f}")
    cols[1].metric("Calibrated Brier · 12m", f"{brier12['cox_calibrated']:.4f}")
    cols[2].metric("Baseline Brier · 12m", f"{brier12['baseline']:.4f}")
    cols[3].metric("Observed test churn", f"{metrics['test']['observed_churn_rate']:.1%}")
    st.info(
        "Train: assigned Dec 2021/2022 landmarks, outcomes through Dec 2023. Calibrate: Dec 2023 → Dec 2024. Test: Dec 2024 → Dec 2025. No test outcomes tune the model or calibrator."
    )
    left, right = st.columns(2)
    with left:
        financial = data["test_financial"]
        overall = financial[financial.dimension == "all"].melt(
            "model", value_vars=["mae", "rmse"], var_name="Metric", value_name="Error"
        )
        plot(
            px.bar(
                overall,
                x="Metric",
                y="Error",
                color="model",
                color_discrete_map=MODEL_COLORS,
                barmode="group",
                title="Financial forecast error · 12 months",
                labels={"Error": "USD per account"},
            )
        )
    with right:
        calibration = data["test_calibration"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 0.4],
                y=[0, 0.4],
                mode="lines",
                line=dict(color="#CBD5E1", dash="dash"),
                name="Perfect calibration",
            )
        )
        for name, group in calibration.groupby("model", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=group.predicted,
                    y=group.observed,
                    mode="markers+lines",
                    name=name,
                    line=dict(color=MODEL_COLORS[name]),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=group.upper - group.observed,
                        arrayminus=group.observed - group.lower,
                    ),
                    customdata=group[["count"]],
                    hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>Accounts %{customdata[0]}<extra>%{fullData.name}</extra>",
                )
            )
        fig.update_layout(
            title="Calibration · 12-month risk buckets",
            xaxis_title="Mean predicted risk",
            yaxis_title="Observed churn fraction",
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        plot(fig)
    st.subheader("Performance by segment")
    dimension = st.selectbox("Breakdown", ["company_size", "product_family", "acquisition_cohort"])
    table = financial[financial.dimension == dimension][
        [
            "segment",
            "model",
            "accounts",
            "mae",
            "rmse",
            "mean_bias",
            "actual_total",
            "forecast_total",
        ]
    ]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            k: st.column_config.NumberColumn(k.replace("_", " ").title(), format="$%,.0f")
            for k in ["mae", "rmse", "mean_bias", "actual_total", "forecast_total"]
        },
    )
    with st.expander("Calibration before/after, survival diagnostics, and limitations"):
        st.write(
            f"Validation-fitted cumulative-hazard multiplier: {metrics['calibration_alpha']:.3f}. Recalibration is applied consistently; all raw and calibrated results are disclosed."
        )
        merged = (
            data["validation_brier"]
            .rename(columns={"brier_score": "validation_brier"})
            .merge(
                brier.rename(columns={"brier_score": "test_brier"}), on=["model", "horizon_months"]
            )
        )
        st.dataframe(merged, hide_index=True, width="stretch")
        km = data["kaplan_meier"]
        plot(
            px.line(km, x="month", y=km.columns[1], title="Training-landmark Kaplan–Meier survival")
        )
        st.write(
            "Brier scores are evaluated only at 3/6/12 months with complete follow-up. Earlier censoring causes an error, rather than being treated as survival. C-index is Harrell's estimator. Cox proportional hazards and static covariates are modeling assumptions; no causal claim or real-world validation is implied. Hardware margins, future sales, acquisition costs, and intervention effects are excluded from CLV."
        )
        st.dataframe(data["cox_coefficients"], hide_index=True, width="stretch")
    st.download_button(
        "Download validation breakdown",
        financial.to_csv(index=False),
        "model_validation.csv",
        "text/csv",
    )
