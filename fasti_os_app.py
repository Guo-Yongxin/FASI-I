# fasti_os_app.py
import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta
import base64
from io import BytesIO
from textwrap import dedent
from PIL import Image


# =========================================
# Config
# =========================================
st.set_page_config(
    page_title="FAST-I HNSCC Overall Survival Risk Calculator",
    page_icon="🩸",
    layout="wide"
)

DEPLOY_DIR = "."  # app.py 放在 FASTI_OS_deploy 文件夹内时，用当前目录


# =========================================
# Load deploy assets
# =========================================
@st.cache_data
def load_assets():
    coef_df = pd.read_csv(f"{DEPLOY_DIR}/coef_os.csv")
    xcols_df = pd.read_csv(f"{DEPLOY_DIR}/x_columns_os.csv")
    base_df = pd.read_csv(f"{DEPLOY_DIR}/baseline_survival_os.csv")
    winsor_df = pd.read_csv(f"{DEPLOY_DIR}/winsor_cutoffs_os.csv")

    with open(f"{DEPLOY_DIR}/model_meta_os.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(f"{DEPLOY_DIR}/lp_cutoffs_os.json", "r", encoding="utf-8") as f:
        lp_cutoffs = json.load(f)

    beta_map = dict(zip(coef_df["term"], coef_df["beta"]))
    x_col_order = xcols_df["x_col_order"].tolist()
    center = float(meta["center"])
    knots = meta["rcs"]["knots"]

    winsor_map = {
        row["variable"]: {"p01": float(row["p01"]), "p99": float(row["p99"])}
        for _, row in winsor_df.iterrows()
    }

    return coef_df, x_col_order, base_df, meta, beta_map, center, knots, winsor_map, lp_cutoffs


coef_df, x_col_order, base_df, meta, beta_map, center, knots, winsor_map, lp_cutoffs = load_assets()

# =========================================
# Helpers
# =========================================
def image_to_base64(image_path: str, crop_box=None) -> str:
    if crop_box is None:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    with Image.open(image_path) as img:
        img = img.convert("RGBA").crop(crop_box)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def normalize_cat(v):
    if pd.isna(v):
        raise ValueError(f"Missing categorical value: {v}")
    fv = float(v)
    if abs(fv - round(fv)) > 1e-12:
        raise ValueError(f"Categorical value is not integer-like: {v}")
    return str(int(round(fv)))


def baseline_survival_at_time(t: float) -> float:
    times = base_df["time"].to_numpy(dtype=float)
    survs = base_df["baseline_survival"].to_numpy(dtype=float)

    if t <= times[0]:
        return float(survs[0])
    if t >= times[-1]:
        return float(survs[-1])

    return float(np.interp(t, times, survs))


def _pos_cube(z: float) -> float:
    return max(z, 0.0) ** 3


def rcs_basis_3knots(x: float, knots_list):
    k1, k2, k3 = [float(v) for v in knots_list]

    b1 = float(x)
    b2 = (
        _pos_cube(x - k1)
        + (
            (k2 - k1) * _pos_cube(x - k3)
            - (k3 - k1) * _pos_cube(x - k2)
        ) / (k3 - k2)
    ) / ((k3 - k1) ** 2)

    return b1, b2

def winsorize_value(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def compute_cleaned_predictors_from_raw(row_raw: dict) -> dict:
    hb_pre = float(row_raw["HB_pre"])
    alb_pre = float(row_raw["ALB_pre"])
    alb_post = float(row_raw["ALB_post"])
    ly_pre = float(row_raw["LY_pre"])
    ly_post = float(row_raw["LY_post"])
    mo_pre = float(row_raw["MO_pre"])
    mo_post = float(row_raw["MO_post"])
    dt = float(row_raw["dt"])

    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if mo_pre <= 0 or mo_post <= 0:
        raise ValueError("MO_pre and MO_post must be > 0.")
    if alb_pre <= 0 or alb_post <= 0:
        raise ValueError("ALB_pre and ALB_post must be > 0.")

    hb_pre_w = winsorize_value(hb_pre, winsor_map["HB_pre"]["p01"], winsor_map["HB_pre"]["p99"])
    alb_pre_w = winsorize_value(alb_pre, winsor_map["ALB_pre"]["p01"], winsor_map["ALB_pre"]["p99"])
    alb_post_w = winsorize_value(alb_post, winsor_map["ALB_post"]["p01"], winsor_map["ALB_post"]["p99"])
    ly_pre_w = winsorize_value(ly_pre, winsor_map["LY_pre"]["p01"], winsor_map["LY_pre"]["p99"])
    ly_post_w = winsorize_value(ly_post, winsor_map["LY_post"]["p01"], winsor_map["LY_post"]["p99"])
    mo_pre_w = winsorize_value(mo_pre, winsor_map["MO_pre"]["p01"], winsor_map["MO_pre"]["p99"])
    mo_post_w = winsorize_value(mo_post, winsor_map["MO_post"]["p01"], winsor_map["MO_post"]["p99"])

    lmr_pre_w = ly_pre_w / mo_pre_w
    lmr_post_w = ly_post_w / mo_post_w
    lmr_a_w = lmr_post_w - lmr_pre_w
    lmr_dt_w = lmr_a_w / dt
    alb_l_w = math.log(alb_post_w) - math.log(alb_pre_w)

    return {
        "HB_pre_w": hb_pre_w,
        "ALB_pre_w": alb_pre_w,
        "ALB_post_w": alb_post_w,
        "LY_pre_w": ly_pre_w,
        "LY_post_w": ly_post_w,
        "MO_pre_w": mo_pre_w,
        "MO_post_w": mo_post_w,
        "LMR_pre_w": lmr_pre_w,
        "LMR_post_w": lmr_post_w,
        "LMR_A_w": lmr_a_w,
        "LMR_dt_w": lmr_dt_w,
        "ALB_L_w": alb_l_w,
    }


def predict_fasti_os_from_raw(row_raw: dict, times=(36.0, 60.0)):
    cleaned = compute_cleaned_predictors_from_raw(row_raw)

    row_cleaned = {
        "p16": row_raw["p16"],
        "Stage0": row_raw["Stage0"],
        "Age": row_raw["Age"],
        "Smoke": row_raw["Smoke"],
        "interval_post": row_raw["interval_post"],
        "LMR_pre_w": cleaned["LMR_pre_w"],
        "ALB_pre_w": cleaned["ALB_pre_w"],
        "ALB_L_w": cleaned["ALB_L_w"],
        "HB_pre_w": cleaned["HB_pre_w"],
        "LMR_dt_w": cleaned["LMR_dt_w"],
    }

    pred = predict_fasti_os_cleaned(row_cleaned, times=times)
    return pred, cleaned, row_cleaned

def build_design_row_cleaned(row: dict) -> pd.DataFrame:
    x = {col: 0.0 for col in x_col_order}

    x["Age"] = float(row["Age"])
    x["interval_post"] = float(row["interval_post"])
    x["LMR_pre_w"] = float(row["LMR_pre_w"])
    x["ALB_pre_w"] = float(row["ALB_pre_w"])
    x["ALB_L_w"] = float(row["ALB_L_w"])
    x["HB_pre_w"] = float(row["HB_pre_w"])

    p16 = normalize_cat(row["p16"])
    stage0 = normalize_cat(row["Stage0"])
    smoke = normalize_cat(row["Smoke"])

    if p16 == "1":
        x["p161"] = 1.0
    elif p16 == "2":
        x["p162"] = 1.0
    elif p16 != "0":
        raise ValueError(f"Invalid p16: {p16}")

    if stage0 == "2":
        x["Stage02"] = 1.0
    elif stage0 != "1":
        raise ValueError(f"Invalid Stage0: {stage0}")

    if smoke == "1":
        x["Smoke1"] = 1.0
    elif smoke == "2":
        x["Smoke2"] = 1.0
    elif smoke != "0":
        raise ValueError(f"Invalid Smoke: {smoke}")

    rcs_col1 = "rcs(LMR_dt_w, c(-0.088823529, -0.037810232, -0.008348214))LMR_dt_w"
    rcs_col2 = "rcs(LMR_dt_w, c(-0.088823529, -0.037810232, -0.008348214))LMR_dt_w'"

    b1, b2 = rcs_basis_3knots(float(row["LMR_dt_w"]), knots)
    x[rcs_col1] = b1
    x[rcs_col2] = b2

    return pd.DataFrame([x], columns=x_col_order)


def predict_fasti_os_cleaned(row: dict, times=(36.0, 60.0)):
    X = build_design_row_cleaned(row)
    beta = np.array([beta_map[t] for t in coef_df["term"]], dtype=float)
    x_vec = X.iloc[0].to_numpy(dtype=float)

    lp_raw = float(np.dot(x_vec, beta))
    lp = lp_raw - center

    out = {"lp": lp}

    for t in times:
        s0 = baseline_survival_at_time(float(t))
        risk_t = 1.0 - (s0 ** math.exp(lp))
        out[f"OSrisk{int(t)}"] = risk_t

    return out


def make_survival_curve(lp: float, max_time: int = 120):
    ts = np.arange(0, max_time + 1, 1, dtype=float)
    surv = []
    risk = []

    for t in ts:
        s0 = baseline_survival_at_time(float(t))
        s = s0 ** math.exp(lp)
        surv.append(s)
        risk.append(1.0 - s)

    return pd.DataFrame({
        "time": ts,
        "survival": surv,
        "risk": risk
    })

def risk_group_from_lp(lp: float) -> str:
    median_cut = float(lp_cutoffs["median"])
    return "High risk" if lp >= median_cut else "Low risk"

def render_centered_table(df: pd.DataFrame, title: str = None, float_format: str = None):
    show_df = df.copy().reset_index(drop=True)

    if float_format is not None:
        for col in show_df.columns:
            if pd.api.types.is_float_dtype(show_df[col]):
                show_df[col] = show_df[col].map(lambda x: format(x, float_format))

    header_html = "".join(
        [f'<th style="text-align:center; padding:10px 8px;">{col}</th>' for col in show_df.columns]
    )

    body_rows = []
    for _, row in show_df.iterrows():
        cells = "".join(
            [f'<td style="text-align:center; padding:10px 8px;">{row[col]}</td>' for col in show_df.columns]
        )
        body_rows.append(f"<tr>{cells}</tr>")

    body_html = "".join(body_rows)

    if title:
        st.markdown(f"**{title}**")

    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 1rem;
        ">
            <table style="
                width: 100%;
                border-collapse: collapse;
                table-layout: auto;
            ">
                <thead style="background: rgba(148, 163, 184, 0.10);">
                    <tr>{header_html}</tr>
                </thead>
                <tbody>
                    {body_html}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================
# UI
# =========================================
logo_path = "assets/FAST-I.png"
logo_crop_box = (120, 110, 1440, 750)   # 裁图范围 (left, upper, right, lower)

logo_b64 = image_to_base64(logo_path, crop_box=logo_crop_box)

st.markdown(dedent(f"""
<style>
.fasti-hero {{
position: relative;
width: 100%;
margin: 0 0 1rem 0;
border: 1px solid rgba(15, 23, 42, 0.10); 
border-radius: 16px;
overflow: hidden;
box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
background: linear-gradient(120deg, #f3f8ff 0%, #edf5ff 52%, #eef6ff 100%);
}}

.fasti-hero::before {{
content: "";
position: absolute;
inset: 0;
background:
    radial-gradient(circle at 78% 32%, rgba(111, 191, 255, 0.14) 0%, rgba(111, 191, 255, 0) 32%),
    linear-gradient(90deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0) 38%);
pointer-events: none;
}}

.fasti-hero-grid {{
position: relative;
z-index: 1;
display: grid;
grid-template-columns: minmax(300px, 0.60fr) minmax(0, 0.40fr);
min-height: clamp(122px, 13vw, 150px);
align-items: stretch;
}}

.fasti-hero-copy {{
display: flex;
flex-direction: column;
justify-content: center;
gap: 0.35rem;
padding: clamp(0.75rem, 1.3vw, 1rem) clamp(0.95rem, 1.6vw, 1.25rem);
background: transparent;
}}

.fasti-hero-title {{
margin: 0;
font-size: clamp(3rem, 4.5vw, 4.5rem);
font-weight: 850;
line-height: 0.95;
letter-spacing: -0.05em;
color: #10213f;
}}

.fasti-hero-subtitle {{
margin: 0;
max-width: 140ch;
font-size: clamp(1.25rem, 2vw, 2rem);
font-weight: 590;
line-height: 1.32;
text-align: justify;
text-justify: inter-word;
color: #31435f;
}}

.fasti-hero-media {{
position: relative;
display: flex;
align-items: center;
justify-content: center;
min-height: clamp(122px, 13vw, 150px);
padding: clamp(0.15rem, 0.45vw, 0.3rem) clamp(0.55rem, 0.95vw, 0.8rem) clamp(0.1rem, 0.35vw, 0.2rem) 0;
background: transparent;
overflow: hidden;
}}

.fasti-hero-media img {{
width: auto;
height: 100%;
max-width: 100%;
max-height: 82%;
display: block;
object-fit: contain;
object-position: center center;
}}

@media (max-width: 960px) {{
.fasti-hero-grid {{
grid-template-columns: 1fr;
}}

.fasti-hero-copy {{
padding-bottom: 1rem;
}}

.fasti-hero-subtitle {{
max-width: none;
}}

.fasti-hero-media {{
min-height: 118px;
padding: 0 0.55rem 0.45rem 0.55rem;
}}
}}
</style>
<div class="fasti-hero">
<div class="fasti-hero-grid">
<div class="fasti-hero-copy">
<div class="fasti-hero-title">FAST - I</div>
<div class="fasti-hero-subtitle">Dynamic Peri-radiotherapy Hematological Biomarkers 
                   for Overall Survival Risk Stratification in Head and Neck Squamous Cell Carcinoma (HNSCC)</div>
</div>
<div class="fasti-hero-media">
<img src="data:image/png;base64,{logo_b64}" alt="FAST-I banner" />
</div>
</div>
</div>
"""), unsafe_allow_html=True)


st.markdown(
    """
    <div style="
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: rgba(168, 85, 247, 0.10);
        border: 1px solid rgba(168, 85, 247, 0.25);
        margin-bottom: 1rem;
    ">
        <div style="font-weight: 800; color: var(--text-color); margin-bottom: 0.55rem;">
            Predicts overall survival risk in HNSCC using the locked combined model prediction engine.
        </div>
        <div style="font-weight: 800; color: var(--text-color);">
            Intended to support risk stratification and post-radiotherapy surveillance. Its outputs should be interpreted alongside clinical assessment by radiation oncologists.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("How dates and sampling intervals are defined", expanded=False):
    c1, c2, c3 = st.columns([3, 6, 3])
    with c2:
        st.image("assets/time interval.png", width=700)

    st.markdown(
        """
        **Intervals**

        - **interval_pre**: the time from the pre-treatment blood sample to the start of radiotherapy.  
        - **interval_post***: the time of the selected post-treatment blood sample relative to the end of radiotherapy. The nearest available post-treatment blood sample was preferentially selected. <br>Otherwise, the nearest available blood sample before the end of radiotherapy was used, for example, the last week.  
        - **Blood sampling interval (dt)**: the time between the selected pre- and post-treatment samples.
        """,
        unsafe_allow_html=True
    )


with st.sidebar:
    st.header("Patient characteristics")
    Age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=60,
        step=1,
        format="%d"
    )

    smoke_map = {
        "Never": "0",
        "Current": "1",
        "Former": "2",
    }

    stage_map = {
        "Stage I-II": "1",
        "Stage III-IV": "2",
    }
    
    p16_map = {
        "Negative": "0",
        "Positive": "1",
        "Not tested": "2",
    }

    smoke_label = st.selectbox("Smoking history", options=list(smoke_map.keys()), index=0)
    stage_label = st.selectbox("TNM stage (AJCC 8th)", options=list(stage_map.keys()), index=1)
    p16_label = st.selectbox(":violet-badge[p16 status*]", options=list(p16_map.keys()), index=1)

    st.caption(
    "**p16 is routinely assessed in oropharyngeal carcinoma, but may not be reported "
    "for other primary subsites. If p16 is unavailable in the pathology report, "
    "please select 'Not tested'.**",
    text_alignment="justify")
    
    Smoke = smoke_map[smoke_label]
    Stage0 = stage_map[stage_label]
    p16 = p16_map[p16_label]
     
    st.markdown("### Radiotherapy (RT) and blood sampling dates")
    c1, c2 = st.columns(2)
    with c1:
        rt_start_date = st.date_input(
            "RT Start date",
            value=date(2020, 1, 1),
            format="DD-MM-YYYY"
        )
    with c2:
        rt_end_date = st.date_input(
            "RT End date",
            value=date(2020, 2, 19),
            format="DD-MM-YYYY"
        )
    c3, c4 = st.columns(2)
    with c3:
        pre_blood_date = st.date_input(
            "Pre-RT Blood test date",
            value=date(2020,1,1),
            format="DD-MM-YYYY"
        )
    with c4:
        post_blood_date = st.date_input(
            "Post-RT Blood test date",
            value=date(2020,2,25),
            format="DD-MM-YYYY"
        )
    interval_post = int((post_blood_date - rt_end_date).days)
    dt = int((post_blood_date - pre_blood_date).days)

    st.caption(
        f"interval_post*: **{interval_post}** days  \n"
        f"Blood sampling interval: **{dt}** days"
    )
    with st.form("fasti_input_form", clear_on_submit=False):  
        st.markdown("### Blood test results")

        st.markdown("### Pre-RT Blood test results (Baseline)")

        c5, c6 = st.columns(2)
        with c5:
            HB_pre = st.number_input("Hemoglobin (g/L)", value=135, step=1, format="%d")
        with c6:
            ALB_pre = st.number_input("Albumin (g/L)", value=42, step=1, format="%d")

        c7, c8 = st.columns(2)
        with c7:
            LY_pre = st.number_input("Lymphocytes (x10^9/L)", value=1.50, step=0.1, format="%.2f")
        with c8:
            MO_pre = st.number_input("Monocytes (x10^9/L)", value=0.80, step=0.1, format="%.2f")

        st.markdown("### Post-RT Blood test results")

        c9, c10 = st.columns(2)
        with c9:
            ALB_post = st.number_input("Albumin (g/L)", value=38, step=1, format="%d")
        with c10:
            st.markdown("&nbsp;", unsafe_allow_html=True)

        c11, c12 = st.columns(2)
        with c11:
            LY_post = st.number_input("Lymphocytes (x10^9/L)", value=1.00, step=0.1, format="%.2f")
        with c12:
            MO_post = st.number_input("Monocytes (x10^9/L)", value=0.50, step=0.1, format="%.2f")
        
        run_model = st.form_submit_button("Calculate risk")

pred = None
cleaned_calc = None
row_cleaned = None
curve_df = None
risk_group_label = None
predict_ok = False

if run_model:
    row_raw = {
        "p16": p16,
        "Stage0": Stage0,
        "Age": Age,
        "Smoke": Smoke,
        "interval_post": interval_post,
        "dt": dt,
        "HB_pre": HB_pre,
        "ALB_pre": ALB_pre,
        "ALB_post": ALB_post,
        "LY_pre": LY_pre,
        "LY_post": LY_post,
        "MO_pre": MO_pre,
        "MO_post": MO_post,
    }

    try:
        pred, cleaned_calc, row_cleaned = predict_fasti_os_from_raw(row_raw, times=(36.0, 60.0))
        curve_df = make_survival_curve(pred["lp"], max_time=120)
        risk_group_label = risk_group_from_lp(pred["lp"])
        predict_ok = True
    except Exception as e:
        pred = None
        cleaned_calc = None
        row_cleaned = None
        curve_df = None
        risk_group_label = None
        predict_ok = False
        st.error(str(e))

if predict_ok:
    risk_color = "#d62728" if risk_group_label == "High risk" else "#1f77b4"

    st.markdown(
        f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;margin-bottom:1rem;">
        <div style="padding:0.9rem 1rem;border:1px solid rgba(120,120,120,0.18);border-radius:14px;background:white;">
            <div style="font-size:0.95rem;color:#64748b;font-weight:700;margin-bottom:0.35rem;">Risk stratification</div>
            <div style="font-size:1.75rem;font-weight:800;color:{risk_color};">{risk_group_label}</div>
        </div>
        <div style="padding:0.9rem 1rem;border:1px solid rgba(120,120,120,0.18);border-radius:14px;background:white;">
            <div style="font-size:0.95rem;color:#64748b;font-weight:700;margin-bottom:0.35rem;">Risk score</div>
            <div style="font-size:1.75rem;font-weight:800;color:#0f172a;">{pred['lp']:.4f}</div>
        </div>
        <div style="padding:0.9rem 1rem;border:1px solid rgba(120,120,120,0.18);border-radius:14px;background:white;">
            <div style="font-size:0.95rem;color:#64748b;font-weight:700;margin-bottom:0.35rem;">OS risk at 3 years</div>
            <div style="font-size:1.75rem;font-weight:800;color:#0f172a;">{pred['OSrisk36'] * 100:.2f}%</div>
        </div>
        <div style="padding:0.9rem 1rem;border:1px solid rgba(120,120,120,0.18);border-radius:14px;background:white;">
            <div style="font-size:0.95rem;color:#64748b;font-weight:700;margin-bottom:0.35rem;">OS risk at 5 years</div>
            <div style="font-size:1.75rem;font-weight:800;color:#0f172a;">{pred['OSrisk60'] * 100:.2f}%</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True
    )

    st.subheader("Individual overall survival curve")

    fig = go.Figure()

    fig.add_trace(
    go.Scatter(
        x=curve_df["time"],
        y=curve_df["survival"],
        mode="lines",
        name="OS probability",
        line=dict(width=8, color=risk_color)
        )
    )

    fig.add_trace(
    go.Scatter(
        x=[36, 60],
        y=[
            1 - pred["OSrisk36"],
            1 - pred["OSrisk60"],
        ],
        mode="markers",
        name="3y / 5y",
        marker=dict(size=15, line=dict(width=3, color="white"), color= "#0f172a")
        )
    )

    fig.update_layout(
    xaxis_title="Months",
    yaxis_title="Overall survival probability",
    yaxis=dict(
        range=[0, 1],
        title_font=dict(size=18),
        tickfont=dict(size=14)
        ),
    xaxis=dict(
        title_font=dict(size=18),
        tickfont=dict(size=14),
        tickmode="linear",
        dtick=12,
        range=[0, 120]
        ),
    template="plotly_white",
    height=560,
    legend=dict(font=dict(size=14))
    )

    st.plotly_chart(fig, width='stretch')

    with st.expander("Show calculation details", expanded=False):
        raw_display_df = pd.DataFrame([{
            "Age": Age,
            "Smoking history": smoke_label,
            "p16": p16_label,
            "TNM Stage": stage_label,           
            "interval_post*": interval_post,
            "dt*": dt,
            "HB_pre": HB_pre,
            "ALB_pre": ALB_pre,
            "LY_pre": LY_pre,
            "MO_pre": MO_pre,
            "ALB_post": ALB_post,
            "LY_post": LY_post,            
            "MO_post": MO_post,
        }])

        derived_display_df = pd.DataFrame([{
            "HB_pre_w*": cleaned_calc["HB_pre_w"],
            "ALB_pre_w": cleaned_calc["ALB_pre_w"],
            "ALB_post_w": cleaned_calc["ALB_post_w"],
            "LY_pre_w": cleaned_calc["LY_pre_w"],
            "LY_post_w": cleaned_calc["LY_post_w"],
            "MO_pre_w": cleaned_calc["MO_pre_w"],
            "MO_post_w": cleaned_calc["MO_post_w"],
            "LMR_pre_w": cleaned_calc["LMR_pre_w"],
            "LMR_post_w": cleaned_calc["LMR_post_w"],
            "∆LMR_A": cleaned_calc["LMR_A_w"],
            "LMR_dt": cleaned_calc["LMR_dt_w"],
            "∆ALB_L": cleaned_calc["ALB_L_w"],
        }])

        model_input_display_df = pd.DataFrame([{
            "Age": row_cleaned["Age"],
            "Smoking history": row_cleaned["Smoke"],
            "p16": row_cleaned["p16"],
            "TNM Stage": row_cleaned["Stage0"],           
            "interval_post": row_cleaned["interval_post"],
            "HB_pre_w": row_cleaned["HB_pre_w"],
            "ALB_pre_w": row_cleaned["ALB_pre_w"],
            "∆ALB_L": row_cleaned["ALB_L_w"],
            "LMR_pre_w": row_cleaned["LMR_pre_w"],
            "LMR_dt": row_cleaned["LMR_dt_w"],
        }])

        output_display_df = pd.DataFrame([{
            "Risk stratification": risk_group_label,
            "Risk score": pred["lp"],
            "OS risk at 3 years": pred["OSrisk36"],
            "OS risk at 5 years": pred["OSrisk60"],
        }])

        render_centered_table(raw_display_df, "Raw inputs")
        st.caption(
        "*, See the 'How dates and sampling intervals are defined' section for details."
    )
        render_centered_table(derived_display_df, "Derived processed predictors", float_format=".4f")
        st.caption(
        "_ w, All blood variables were winsorised at the 1st and 99th percentiles to limit the influence of extreme outliers (if required)."
    )
        render_centered_table(model_input_display_df, "Prediction inputs", float_format=".4f")
        render_centered_table(output_display_df, "Prediction outputs", float_format=".4f")
else:
    st.info("⬅︎ Please correct the input values and click 'Calculate risk' to generate prediction.")

st.markdown(
    """
    <hr style="margin: 2rem 0 1rem 0; border: none; border-top: 1px solid rgba(120,120,120,0.18);">

    <div style="
        font-size: 0.92rem;
        line-height: 1.6;
        color: var(--text-color);
        opacity: 0.88;
        margin-bottom: 1.2rem;
    ">
        <div style="font-weight: 800; margin-bottom: 0.25rem;">
            FAST–I
        </div>
        <div>
            Developed by <b>Yongxin Guo, Dorothy Gujral, Eric O. Aboagye, Matthew Williams et al.</b>
        </div>
        <div>
            Department of Radiotherapy / Imperial College Healthcare NHS Trust <br>Department of Surgery and Cancer / Imperial College London
        </div>
        <div>
            Model version: <b>FAST–I v1.0</b>
        </div>
        <div>
            Contact: <b>[@nhs.net]</b>
        </div>
        <div style="margin-top: 0.35rem;">
            <b>For research and clinical reference only. Predictions should be interpreted alongside clinical assessment by radiation oncologists.</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
