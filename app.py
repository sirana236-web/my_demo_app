#!/usr/bin/env python3
"""
🔮 Customer Churn Prediction — Live Demo (v2 — Fixed)
======================================================
اجرا:
    streamlit run app.py

وابستگی‌ها:
    pip install streamlit pandas numpy scikit-learn joblib plotly
"""

# ════════════════════════════════════════════════════════════════
#  IMPORTS
# ════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
import hashlib
import json
from pathlib import Path

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
#  VERSION TRACKING — حل مشکل ناسازگاری نسخه
# ════════════════════════════════════════════════════════════════

import sklearn
SKLEARN_VERSION = sklearn.__version__

def _get_version_file() -> Path:
    return Path("models") / "_sklearn_version.txt"

def _is_version_compatible() -> bool:
    """بررسی اینکه مدل‌ها با نسخه فعلی scikit-learn سازگار هستند."""
    vf = _get_version_file()
    if not vf.exists():
        return False
    saved_version = vf.read_text().strip()
    # فقط major.minor باید یکی باشد (مثلاً 1.3 == 1.3)
    current_major_minor = ".".join(SKLEARN_VERSION.split(".")[:2])
    saved_major_minor   = ".".join(saved_version.split(".")[:2])
    return current_major_minor == saved_major_minor

def _save_version():
    _get_version_file().write_text(SKLEARN_VERSION)

# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════

DATA_PATH    = "test_data.csv"
MODELS_DIR   = "models"
HIGH_THRESH  = 0.70
MED_THRESH   = 0.40

NUMERIC_FEATS = [
    "age", "tenure_months", "monthly_revenue", "total_spend",
    "num_support_tickets", "monthly_usage_hours", "num_products",
    "satisfaction_score", "last_login_days",
]
CATEGORICAL_FEATS = ["contract_type", "payment_method"]
ALL_FEATS = NUMERIC_FEATS + CATEGORICAL_FEATS

# ────────── Page Config ──────────
st.set_page_config(
    page_title="Churn Prediction ∙ Live Demo",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── مخفی‌سازی منو، فوتر، دانلود ── */
#MainMenu, footer, header,
[data-testid="stElementToolbar"],
.stDeployButton,
[data-testid="stBaseButton-headerNoPadding"],
button[title="Download"],
[data-testid="StyledFullScreenButton"] {
    display: none !important;
    visibility: hidden !important;
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid #2a2a50;
    border-radius: 18px;
    padding: 26px 18px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-7px);
    box-shadow: 0 14px 40px rgba(0,0,0,0.55);
}
.mc-icon  { font-size: 2.2rem; margin-bottom: 6px; }
.mc-label {
    font-size: 0.78rem; color: #7a7fa0;
    text-transform: uppercase; letter-spacing: 2px;
}
.mc-value { font-size: 2.3rem; font-weight: 800; margin-top: 4px; }

.c-blue   { color: #6C63FF; }
.c-red    { color: #FF4B4B; }
.c-yellow { color: #FFB020; }
.c-green  { color: #00CC66; }

.risk-bar {
    display: flex; height: 10px;
    border-radius: 6px; overflow: hidden; margin: 8px 0 4px 0;
}
.risk-bar > div { transition: width 0.6s ease; }

.sep {
    height: 2px; border: none; margin: 28px 0;
    background: linear-gradient(90deg, transparent, #6C63FF44, transparent);
}

.sub {
    text-align: center; color: #7a7fa0;
    margin-top: -10px; margin-bottom: 26px;
    font-size: 1.05rem;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SETUP — ساخت داده + مدل (فقط اگر موجود نباشد یا نسخه عوض شده)
# ════════════════════════════════════════════════════════════════

def _make_test_csv(n: int = 250) -> pd.DataFrame:
    """ساخت test_data.csv نمونه."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "user_id":              [f"USR-{i:04d}" for i in range(1, n + 1)],
        "age":                  rng.integers(18, 68, n),
        "tenure_months":        rng.integers(1, 72, n),
        "monthly_revenue":      np.round(rng.uniform(15, 550, n), 2),
        "total_spend":          np.round(rng.uniform(200, 30_000, n), 2),
        "num_support_tickets":  rng.integers(0, 18, n),
        "monthly_usage_hours":  np.round(rng.uniform(1, 220, n), 1),
        "num_products":         rng.integers(1, 6, n),
        "satisfaction_score":   rng.integers(1, 6, n),
        "last_login_days":      rng.integers(0, 90, n),
        "contract_type":        rng.choice(
            ["Monthly", "Quarterly", "Annual"], n, p=[0.50, 0.30, 0.20]),
        "payment_method":       rng.choice(
            ["Credit Card", "Bank Transfer", "Digital Wallet"], n,
            p=[0.40, 0.35, 0.25]),
    })
    df.to_csv(DATA_PATH, index=False)
    return df


def _purge_old_models():
    """پاک‌سازی کامل پوشه models/ قبل از ساخت مجدد."""
    import shutil
    if os.path.exists(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR, exist_ok=True)


def _make_models():
    """ساخت مدل‌های Pipeline و ذخیره — با نسخه فعلی scikit-learn."""
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    _purge_old_models()

    # ── Preprocessor ──
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATS),
            ("cat", OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
            ), CATEGORICAL_FEATS),
        ],
        remainder="drop",
    )

    # ── داده آموزشی مصنوعی ──
    rng = np.random.default_rng(99)
    N   = 2_000
    X_train = pd.DataFrame({
        "age":                  rng.integers(18, 68, N),
        "tenure_months":        rng.integers(1, 72, N),
        "monthly_revenue":      np.round(rng.uniform(15, 550, N), 2),
        "total_spend":          np.round(rng.uniform(200, 30_000, N), 2),
        "num_support_tickets":  rng.integers(0, 18, N),
        "monthly_usage_hours":  np.round(rng.uniform(1, 220, N), 1),
        "num_products":         rng.integers(1, 6, N),
        "satisfaction_score":   rng.integers(1, 6, N),
        "last_login_days":      rng.integers(0, 90, N),
        "contract_type":        rng.choice(
            ["Monthly", "Quarterly", "Annual"], N),
        "payment_method":       rng.choice(
            ["Credit Card", "Bank Transfer", "Digital Wallet"], N),
    })

    # ── برچسب مصنوعی ──
    score = (
        0.22 * (X_train["num_support_tickets"] / 18)
        + 0.20 * (1 - X_train["tenure_months"] / 72)
        + 0.15 * (1 - X_train["monthly_usage_hours"] / 220)
        + 0.13 * (1 - X_train["satisfaction_score"] / 5)
        + 0.12 * (X_train["last_login_days"] / 90)
        + 0.08 * (X_train["contract_type"] == "Monthly").astype(float)
        + 0.10 * rng.uniform(0, 1, N)
    )
    y_train = (score > 0.48).astype(int)

    # ── ساخت + ذخیره هر مدل ──
    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=5, random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1_000, random_state=42,
        ),
    }

    for name, clf in classifiers.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   clf),
        ])
        pipe.fit(X_train[ALL_FEATS], y_train)

        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(pipe, path, protocol=4)   # protocol=4 → سازگاری بیشتر

    # ── ذخیره تنظیمات ──
    cfg = {
        "numeric":     NUMERIC_FEATS,
        "categorical": CATEGORICAL_FEATS,
        "all":         ALL_FEATS,
    }
    joblib.dump(cfg, os.path.join(MODELS_DIR, "feature_config.pkl"), protocol=4)

    # ── ثبت نسخه ──
    _save_version()

    return True


def _ensure_models_ready():
    """
    اطمینان از وجود مدل‌ها و سازگاری نسخه.
    اگر نسخه عوض شده → پاک + ساخت مجدد.
    """
    models_exist = (
        os.path.exists(MODELS_DIR)
        and len(list(Path(MODELS_DIR).glob("*.pkl"))) >= 3
    )

    if not models_exist or not _is_version_compatible():
        return _make_models()
    return True


# ════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        _make_test_csv()
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_models() -> tuple:
    """بارگذاری مدل‌ها — با مدیریت خطای نسخه."""

    _ensure_models_ready()

    models = {}
    for p in sorted(Path(MODELS_DIR).glob("*.pkl")):
        if p.stem in ("feature_config", "_sklearn_version"):
            continue
        try:
            models[p.stem] = joblib.load(p)
        except (AttributeError, ModuleNotFoundError, ImportError) as e:
            # ⚠️ اگر هنوز خطای نسخه بود → پاک + ساخت مجدد
            st.warning(f"⚠️ مدل `{p.stem}` ناسازگار بود، بازسازی می‌شود...")
            _make_models()
            # بارگذاری مجدد
            models = {}
            for pp in sorted(Path(MODELS_DIR).glob("*.pkl")):
                if pp.stem in ("feature_config", "_sklearn_version"):
                    continue
                models[pp.stem] = joblib.load(pp)
            break

    cfg_path = Path(MODELS_DIR) / "feature_config.pkl"
    cfg = joblib.load(cfg_path) if cfg_path.exists() else {
        "numeric": NUMERIC_FEATS,
        "categorical": CATEGORICAL_FEATS,
        "all": ALL_FEATS,
    }
    return models, cfg


def risk_label(prob: float) -> str:
    if prob >= HIGH_THRESH:
        return "High"
    if prob >= MED_THRESH:
        return "Medium"
    return "Low"


def run_prediction(model, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """اجرای پیش‌بینی — Pipeline خودش پیش‌پردازش می‌کند."""
    feat_cols = cfg.get("all", ALL_FEATS)
    X = df[feat_cols].copy()
    probs = model.predict_proba(X)[:, 1]
    return probs


# ════════════════════════════════════════════════════════════════
#  UI HELPERS
# ════════════════════════════════════════════════════════════════

def _card(icon, label, value, css_color):
    st.markdown(f"""
    <div class="metric-card">
        <div class="mc-icon">{icon}</div>
        <div class="mc-label">{label}</div>
        <div class="mc-value {css_color}">{value}</div>
    </div>""", unsafe_allow_html=True)


def render_summary(total, n_high, n_med, n_low, rev_at_risk):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: _card("👥", "Total Customers",  f"{total:,}",            "c-blue")
    with c2: _card("🚨", "High Risk",        f"{n_high:,}",           "c-red")
    with c3: _card("⚠️",  "Medium Risk",      f"{n_med:,}",            "c-yellow")
    with c4: _card("✅", "Low Risk",         f"{n_low:,}",            "c-green")
    with c5: _card("💰", "Revenue at Risk",  f"${rev_at_risk:,.0f}",  "c-red")


def render_risk_bar(h_pct, m_pct, l_pct):
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;
                font-size:.82rem;margin-top:20px">
        <span class="c-red">●  High  {h_pct:.1f}%</span>
        <span class="c-yellow">●  Medium  {m_pct:.1f}%</span>
        <span class="c-green">●  Low  {l_pct:.1f}%</span>
    </div>
    <div class="risk-bar">
        <div style="width:{h_pct}%;background:#FF4B4B"></div>
        <div style="width:{m_pct}%;background:#FFB020"></div>
        <div style="width:{l_pct}%;background:#00CC66"></div>
    </div>""", unsafe_allow_html=True)


def styled_results(df: pd.DataFrame):
    """استایل‌دهی جدول نتایج — سازگار با هر نسخه pandas."""

    def _risk_style(v):
        colors = {
            "High":   "background:#FF4B4B30;color:#FF4B4B;font-weight:700",
            "Medium": "background:#FFB02030;color:#FFB020;font-weight:700",
            "Low":    "background:#00CC6630;color:#00CC66;font-weight:700",
        }
        return colors.get(v, "")

    def _prob_style(v):
        if v >= HIGH_THRESH:
            return "color:#FF4B4B;font-weight:700"
        if v >= MED_THRESH:
            return "color:#FFB020;font-weight:700"
        return "color:#00CC66;font-weight:700"

    styler = df.style

    # ── سازگاری pandas ≥2.1 و <2.1 ──
    if hasattr(styler, "map"):
        styler = styler.map(_risk_style, subset=["risk_level"])
        styler = styler.map(_prob_style, subset=["churn_probability"])
    else:
        styler = styler.applymap(_risk_style, subset=["risk_level"])
        styler = styler.applymap(_prob_style, subset=["churn_probability"])

    styler = styler.format({
        "churn_probability": "{:.1%}",
        "monthly_revenue":   "${:,.2f}",
    })
    return styler


def render_charts(results: pd.DataFrame):
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.info("📦  `pip install plotly` برای نمایش نمودار")
        return

    col_a, col_b = st.columns(2)

    with col_a:
        counts = results["risk_level"].value_counts().reindex(
            ["High", "Medium", "Low"], fill_value=0)
        fig1 = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            hole=0.55,
            marker_colors=["#FF4B4B", "#FFB020", "#00CC66"],
            textinfo="label+percent",
            textfont_size=13,
        ))
        fig1.update_layout(
            title_text="Risk Distribution", title_x=0.5,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", height=340,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.histogram(
            results, x="churn_probability", nbins=30,
            color_discrete_sequence=["#6C63FF"],
        )
        fig2.add_vline(x=HIGH_THRESH, line_dash="dash",
                       line_color="#FF4B4B", annotation_text="High")
        fig2.add_vline(x=MED_THRESH, line_dash="dash",
                       line_color="#FFB020", annotation_text="Medium")
        fig2.update_layout(
            title_text="Probability Distribution", title_x=0.5,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", height=340,
            margin=dict(t=50, b=20, l=20, r=20),
            xaxis_title="Churn Probability",
            yaxis_title="Count",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    # ──── HEADER ────
    st.markdown(
        "<h1 style='text-align:center;"
        "background:linear-gradient(135deg,#6C63FF,#E040FB);"
        "-webkit-background-clip:text;"
        "-webkit-text-fill-color:transparent;"
        "font-size:2.6rem;font-weight:900;"
        "padding:12px 0'>"
        "🔮  Customer Churn Prediction System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='sub'>Real‑time churn risk analysis · "
        "Powered by Machine Learning</p>",
        unsafe_allow_html=True,
    )

    # ── نمایش نسخه scikit-learn ──
    st.markdown(
        f"<p style='text-align:center;color:#555;font-size:.75rem'>"
        f"scikit-learn v{SKLEARN_VERSION} &nbsp;·&nbsp; "
        f"Python {sys.version.split()[0]}</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    # ──── LOAD ────
    with st.spinner("⏳  Loading data & models …"):
        data = load_data()
        models, cfg = load_models()

    if not models:
        st.error("❌  هیچ مدلی بارگذاری نشد! پوشه `models/` را بررسی کنید.")
        st.stop()

    # ──── MODEL SELECTOR ────
    st.markdown("#### ⚙️  Model Selection")
    pretty = {k: k.replace("_", " ").title() for k in models}

    col_s, col_i = st.columns([2, 3])
    with col_s:
        sel = st.selectbox("Choose a model:", list(models.keys()),
                           format_func=lambda x: pretty[x])
    with col_i:
        m = models[sel]
        if hasattr(m, "named_steps"):
            steps = " → ".join(m.named_steps.keys())
            st.info(f"🔗 **Pipeline:** `{steps}`")
        else:
            st.info(f"📦 Model: `{type(m).__name__}`")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    # ──── PREDICT ────
    try:
        probs = run_prediction(models[sel], data, cfg)
    except Exception as exc:
        st.error(f"❌ خطا در پیش‌بینی: {exc}")
        # یک بار بازسازی
        st.warning("🔄 بازسازی مدل‌ها ...")
        _make_models()
        st.cache_resource.clear()
        st.rerun()

    results = pd.DataFrame({
        "user_id":           data["user_id"],
        "churn_probability": probs,
        "risk_level":        [risk_label(p) for p in probs],
        "monthly_revenue":   data["monthly_revenue"],
    }).sort_values("churn_probability", ascending=False).reset_index(drop=True)

    # ──── METRICS ────
    total   = len(results)
    high_df = results[results.risk_level == "High"]
    med_df  = results[results.risk_level == "Medium"]
    low_df  = results[results.risk_level == "Low"]
    n_high, n_med, n_low = len(high_df), len(med_df), len(low_df)
    rev_risk  = high_df["monthly_revenue"].sum()
    total_rev = results["monthly_revenue"].sum()
    h_pct = n_high / total * 100
    m_pct = n_med  / total * 100
    l_pct = n_low  / total * 100

    # ──── DASHBOARD ────
    st.markdown("### 📊  Dashboard Summary")
    render_summary(total, n_high, n_med, n_low, rev_risk)
    render_risk_bar(h_pct, m_pct, l_pct)
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    # ──── CHARTS ────
    st.markdown("### 📈  Visual Overview")
    render_charts(results)
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    # ──── FILTERS ────
    st.markdown("### 🔍  Filter & Explore")
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        risk_filter = st.multiselect(
            "Risk Level:", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
    with f2:
        top_n = st.slider("Show Top N:", 10, total, min(50, total), 10)
    with f3:
        sort_col = st.selectbox(
            "Sort by:",
            ["churn_probability", "monthly_revenue"],
            format_func=lambda x: x.replace("_", " ").title(),
        )

    view = (
        results[results.risk_level.isin(risk_filter)]
        .sort_values(sort_col, ascending=False)
        .head(top_n)
    )

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    # ──── TABLE ────
    st.markdown(
        f"### 📋  Prediction Results — "
        f"Showing **{len(view):,}** of {total:,}")

    st.dataframe(
        styled_results(view),
        use_container_width=True,
        height=520,
        hide_index=True,
    )

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    # ──── TOP 5 HIGH RISK ────
    st.markdown("### 🚨  Top 5 High‑Risk Customers")
    top5 = high_df.head(5)
    if top5.empty:
        st.success("✅ هیچ مشتری High Risk یافت نشد!")
    else:
        cols = st.columns(len(top5))
        for i, (_, row) in enumerate(top5.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="border-color:#FF4B4B55">
                    <div style="font-size:1.1rem;font-weight:700;
                                color:#FF4B4B;margin-bottom:8px">
                        {row.user_id}
                    </div>
                    <div style="font-size:.8rem;color:#8892b0">
                        Churn: <b style="color:#FF4B4B">
                        {row.churn_probability:.1%}</b>
                    </div>
                    <div style="font-size:.8rem;color:#8892b0;margin-top:4px">
                        Revenue: <b style="color:#FFB020">
                        ${row.monthly_revenue:,.2f}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    # ──── DETAILED STATS ────
    with st.expander("📊  Detailed Statistics", expanded=False):
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("##### Churn Probability")
            st.metric("Mean",   f"{results.churn_probability.mean():.1%}")
            st.metric("Median", f"{results.churn_probability.median():.1%}")
            st.metric("Std",    f"{results.churn_probability.std():.1%}")
        with s2:
            st.markdown("##### Revenue")
            st.metric("Total Revenue",   f"${total_rev:,.0f}")
            st.metric("Revenue at Risk", f"${rev_risk:,.0f}")
            ratio = (rev_risk / total_rev * 100) if total_rev else 0
            st.metric("Risk Ratio",      f"{ratio:.1f}%")
        with s3:
            st.markdown("##### Distribution")
            st.metric("High",   f"{n_high:,}  ({h_pct:.1f}%)")
            st.metric("Medium", f"{n_med:,}  ({m_pct:.1f}%)")
            st.metric("Low",    f"{n_low:,}  ({l_pct:.1f}%)")

    # ──── FOOTER ────
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#555;font-size:.82rem;padding:10px 0'>"
        "🔮 Churn Prediction v2.0 · Fixed Edition</p>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()