"""
KSP Accident Severity Prediction — Streamlit app.

Usage:
    streamlit run app.py

Requires model.cbm and model_meta.json (run train.py first).
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

MODEL_FILE = "model.cbm"
META_FILE = "model_meta.json"
DATA_FILE = "processed_data.csv"

MONTH_LABELS = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

DEPLOYMENT_INFO = {
    "Fatal": {
        "level": 2,
        "label": "High Deployment",
        "message": "Multiple units required — ambulance, police, and traffic control.",
        "border": "#ef4444",
        "text": "#ef4444",
    },
    "Non-Fatal": {
        "level": 1,
        "label": "Standard Response",
        "message": "Single police unit sufficient. Ambulance on standby.",
        "border": "#22c55e",
        "text": "#22c55e",
    },
}

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    .result-card {
        border-left: 4px solid;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        background: var(--secondary-background-color);
    }
    .result-card .rc-label {
        font-size: 0.73rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-color);
        opacity: 0.55;
        margin-bottom: 0.3rem;
    }
    .result-card .rc-value {
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .result-card .rc-sub {
        font-size: 0.83rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-top: 0.25rem;
    }

    .placeholder-box {
        border: 2px dashed;
        border-color: var(--text-color);
        opacity: 0.2;
        border-radius: 10px;
        padding: 2.5rem 1rem;
        text-align: center;
        color: var(--text-color);
        font-size: 0.88rem;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .stat-box .sv { font-size: 1.45rem; font-weight: 700; color: var(--primary-color); }
    .stat-box .sl { font-size: 0.72rem; color: var(--text-color); opacity: 0.6;
                    margin-top: 0.1rem; text-transform: uppercase; letter-spacing: 0.04em; }

    .section-hint {
        font-size: 0.78rem;
        color: var(--text-color);
        opacity: 0.5;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: -0.4rem;
    }
</style>
"""

CHART_CONFIG = {
    "config": {
        "axis": {"gridColor": "#334155", "labelColor": "#94a3b8", "titleColor": "#94a3b8"},
        "view": {"strokeWidth": 0},
        "background": "transparent",
    }
}


@st.cache_resource
def load_model() -> tuple[CatBoostClassifier, dict]:
    model = CatBoostClassifier()
    model.load_model(MODEL_FILE)
    with open(META_FILE) as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def chart(spec: alt.Chart) -> dict:
    """Convert Altair spec to Vega-Lite dict with shared dark theme config."""
    d = spec.to_dict()
    d.update(CHART_CONFIG)
    return d


def result_card(label: str, value: str, sub: str, border: str, text: str) -> None:
    st.markdown(
        f'<div class="result-card" style="border-left-color:{border}">'
        f'  <div class="rc-label">{label}</div>'
        f'  <div class="rc-value" style="color:{text}">{value}</div>'
        f'  <div class="rc-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def predict_tab(model: CatBoostClassifier, meta: dict) -> None:
    st.markdown(
        "Select accident context below. The model predicts injury severity "
        "and recommends an emergency deployment level."
    )
    st.divider()

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<p class="section-hint">Accident Context</p>', unsafe_allow_html=True)
        district = st.selectbox("District", meta["districts"])
        road_type = st.selectbox("Road Type", meta["road_types"])
        col_m, col_y = st.columns(2)
        with col_m:
            month_label = st.selectbox("Month", list(MONTH_LABELS.values()))
        with col_y:
            year = st.number_input("Year", min_value=2016, max_value=2025, value=2022, step=1)
        clicked = st.button("Run Prediction", type="primary", use_container_width=True)

    with right:
        st.markdown('<p class="section-hint">Result</p>', unsafe_allow_html=True)
        if clicked:
            month = next(k for k, v in MONTH_LABELS.items() if v == month_label)
            input_df = pd.DataFrame([{
                "District_Name": district,
                "Road_Type": road_type,
                "Month": month,
                "Year": int(year),
            }])
            prediction = str(model.predict(input_df)[0])
            info = DEPLOYMENT_INFO.get(prediction, DEPLOYMENT_INFO["Non-Fatal"])

            result_card(
                "Predicted Severity",
                prediction,
                f"{district}  ·  {road_type}  ·  {month_label} {year}",
                info["border"], info["text"],
            )
            result_card(
                f"Deployment Level {info['level']} — {info['label']}",
                info["message"],
                "Based on historical KSP accident data",
                info["border"], info["text"],
            )
        else:
            st.markdown(
                '<div class="placeholder-box">'
                'Fill in the inputs and click<br><strong>Run Prediction</strong>'
                '</div>',
                unsafe_allow_html=True,
            )


def insights_tab(df: pd.DataFrame) -> None:
    total = len(df)
    fatal = int((df["Severity"] == "Fatal").sum())
    fatal_pct = fatal / total * 100
    districts = df["District_Name"].nunique()
    years = f"{df['Year'].min()}–{df['Year'].max()}"

    st.markdown(
        f'<div class="stat-grid">'
        f'<div class="stat-box"><div class="sv">{total:,}</div><div class="sl">Total Accidents</div></div>'
        f'<div class="stat-box"><div class="sv">{fatal:,}</div><div class="sl">Fatal Cases</div></div>'
        f'<div class="stat-box"><div class="sv">{fatal_pct:.1f}%</div><div class="sl">Fatality Rate</div></div>'
        f'<div class="stat-box"><div class="sv">{districts}</div><div class="sl">Districts</div></div>'
        f'<div class="stat-box"><div class="sv">{years}</div><div class="sl">Year Range</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("Accidents by Month")
        monthly = df.groupby("Month").size().reset_index(name="Count")
        monthly["Mon"] = monthly["Month"].map(lambda x: MONTH_LABELS.get(x, str(x))[:3])
        st.vega_lite_chart(
            chart(
                alt.Chart(monthly).mark_bar(
                    cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#3b82f6"
                ).encode(
                    x=alt.X("Mon:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Count:Q", title="Accidents"),
                    tooltip=[alt.Tooltip("Mon:N", title="Month"),
                             alt.Tooltip("Count:Q", format=",")],
                ).properties(height=240)
            ),
            use_container_width=True,
        )

    with col2:
        st.subheader("Severity Split")
        nonfatal_n = total - fatal
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(
                f'<div style="background:var(--secondary-background-color);border-left:4px solid #ef4444;'
                f'border-radius:8px;padding:1.1rem 1.2rem;margin-top:0.5rem">'
                f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;opacity:0.55">Fatal</div>'
                f'<div style="font-size:2rem;font-weight:700;color:#ef4444">{fatal:,}</div>'
                f'<div style="font-size:0.8rem;opacity:0.6">{fatal_pct:.1f}% of total</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c_b:
            st.markdown(
                f'<div style="background:var(--secondary-background-color);border-left:4px solid #22c55e;'
                f'border-radius:8px;padding:1.1rem 1.2rem;margin-top:0.5rem">'
                f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;opacity:0.55">Non-Fatal</div>'
                f'<div style="font-size:2rem;font-weight:700;color:#22c55e">{nonfatal_n:,}</div>'
                f'<div style="font-size:0.8rem;opacity:0.6">{nonfatal_n/total*100:.1f}% of total</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    col3, col4 = st.columns(2, gap="medium")

    with col3:
        st.subheader("Top 10 Districts")
        top_d = df["District_Name"].value_counts().head(10).reset_index()
        top_d.columns = ["District", "Count"]
        st.vega_lite_chart(
            chart(
                alt.Chart(top_d).mark_bar(
                    cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#8b5cf6"
                ).encode(
                    x=alt.X("Count:Q", title="Accidents"),
                    y=alt.Y("District:N", sort="-x", title=None),
                    tooltip=[alt.Tooltip("District:N"),
                             alt.Tooltip("Count:Q", format=",")],
                ).properties(height=300)
            ),
            use_container_width=True,
        )

    with col4:
        st.subheader("Accidents by Road Type")
        road = df["Road_Type"].value_counts().reset_index()
        road.columns = ["Road_Type", "Count"]
        st.vega_lite_chart(
            chart(
                alt.Chart(road).mark_bar(
                    cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#f59e0b"
                ).encode(
                    x=alt.X("Road_Type:N", title=None, axis=alt.Axis(labelAngle=-15)),
                    y=alt.Y("Count:Q", title="Accidents"),
                    tooltip=[alt.Tooltip("Road_Type:N", title="Road Type"),
                             alt.Tooltip("Count:Q", format=",")],
                ).properties(height=300)
            ),
            use_container_width=True,
        )

    st.subheader("Fatality Rate by Year")
    yearly = (
        df.groupby("Year")[["Severity"]]
        .apply(lambda g: pd.Series({
            "Total": len(g),
            "Fatal": int((g["Severity"] == "Fatal").sum()),
        }))
        .reset_index()
    )
    yearly["Rate"] = (yearly["Fatal"] / yearly["Total"] * 100).round(2)
    st.vega_lite_chart(
        chart(
            alt.Chart(yearly).mark_line(
                point=alt.OverlayMarkDef(color="#ef4444", size=60),
                color="#ef4444",
                strokeWidth=2,
            ).encode(
                x=alt.X("Year:O", title="Year"),
                y=alt.Y("Rate:Q", title="Fatality Rate (%)"),
                tooltip=[
                    alt.Tooltip("Year:O"),
                    alt.Tooltip("Total:Q", format=",", title="Total"),
                    alt.Tooltip("Fatal:Q", format=",", title="Fatal"),
                    alt.Tooltip("Rate:Q", format=".2f", title="Rate (%)"),
                ],
            ).properties(height=220)
        ),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="KSP Accident Analysis",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("KSP Accident Severity Prediction")
    st.caption("Karnataka State Police  ·  Motor Vehicle Accident Analysis  ·  2016–2024")

    if not Path(MODEL_FILE).exists() or not Path(META_FILE).exists():
        st.error("Model files not found. Run `python3 preprocess.py` then `python3 train.py` first.")
        return

    model, meta = load_model()
    data_ok = Path(DATA_FILE).exists()

    tab1, tab2 = st.tabs(["Predict", "Insights"])

    with tab1:
        predict_tab(model, meta)

    with tab2:
        if not data_ok:
            st.warning("processed_data.csv not found. Insights tab unavailable.")
        else:
            df = load_data()
            insights_tab(df)


if __name__ == "__main__":
    main()
