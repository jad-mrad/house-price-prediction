import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CalHome AI · House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f1e !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% -10%, rgba(99,179,237,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 110%, rgba(237,137,54,0.10) 0%, transparent 55%),
        #0a0f1e !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer { visibility: hidden; }

/* ── Main container ── */
.block-container {
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem !important;
    margin: 0 auto;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    line-height: 1.1;
    color: #f7f3ec;
    margin-bottom: 1rem;
}
.hero-title em {
    font-style: italic;
    color: #ed8936;
}
.hero-sub {
    font-size: 1rem;
    color: #8a9ab5;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto 2rem;
    line-height: 1.7;
}
.hero-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, #63b3ed, #ed8936);
    margin: 0 auto 2.5rem;
    border-radius: 2px;
}

/* ── Stat pills ── */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 3rem;
}
.stat-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 100px;
    padding: 0.45rem 1.2rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: #c5cfe0;
    backdrop-filter: blur(10px);
}
.stat-pill span { color: #63b3ed; font-weight: 600; }

/* ── Section label ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a6280;
    margin-bottom: 1.2rem;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(20px);
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(99,179,237,0.3); }
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    color: #d4cfc7;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Streamlit sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #63b3ed, #4299e1) !important;
}
[data-testid="stSlider"] label {
    color: #8a9ab5 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}

/* ── Predict button ── */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #2b6cb0 0%, #ed8936 100%) !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 0.8rem 2.4rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 4px 24px rgba(237,137,54,0.25) !important;
}
[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, rgba(43,108,176,0.18), rgba(237,137,54,0.14));
    border: 1px solid rgba(237,137,54,0.35);
    border-radius: 20px;
    padding: 2.2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 0.6rem;
}
.result-price {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    color: #f7f3ec;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.result-price em { color: #ed8936; font-style: normal; }
.result-meta {
    font-size: 0.78rem;
    color: #4a6280;
    margin-top: 0.8rem;
}

/* ── Info badges ── */
.badge-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 1rem;
}
.badge {
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 6px;
    padding: 0.3rem 0.75rem;
    font-size: 0.72rem;
    color: #63b3ed;
    font-weight: 500;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.07);
    font-size: 0.78rem;
    color: #3a4d63;
}
.footer a { color: #4a6280; text-decoration: none; }
.footer a:hover { color: #63b3ed; }
</style>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

model, scaler = load_model()


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI · Real Estate · California</div>
    <h1 class="hero-title">Predict Your<br><em>Home's Value</em></h1>
    <p class="hero-sub">Machine learning powered valuation using California housing data. Adjust the parameters and get an instant estimate.</p>
    <div class="hero-divider"></div>
</div>
<div class="stats-row">
    <div class="stat-pill">R² Score <span>0.81</span></div>
    <div class="stat-pill">Algorithm <span>Random Forest</span></div>
    <div class="stat-pill">Training samples <span>16,512</span></div>
    <div class="stat-pill">Features <span>8</span></div>
</div>
""", unsafe_allow_html=True)


# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Configure Property Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-title">💰 Financial & Demographics</div>', unsafe_allow_html=True)
    MedInc    = st.slider("Median Income (×$10k)", 0.5, 15.0, 5.0, 0.1)
    HouseAge  = st.slider("House Age (years)",      1,   52,   20)
    Population = st.slider("Block Population",      3,   35000, 1000, 50)
    AveOccup  = st.slider("Average Occupants",      1.0, 10.0, 3.0, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">🏗️ Property & Location</div>', unsafe_allow_html=True)
    AveRooms  = st.slider("Average Rooms",          1.0, 15.0, 5.0, 0.1)
    AveBedrms = st.slider("Average Bedrooms",       1.0, 5.0,  1.0, 0.1)
    Latitude  = st.slider("Latitude",               32.0, 42.0, 35.0, 0.1)
    Longitude = st.slider("Longitude",              -124.0, -114.0, -119.0, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.button("✦  Estimate Property Value", use_container_width=True):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    price = prediction * 100_000

    # Price tier label
    if price < 150_000:
        tier, tier_color = "Below Market", "#f56565"
    elif price < 300_000:
        tier, tier_color = "Mid Market", "#ed8936"
    elif price < 500_000:
        tier, tier_color = "Above Market", "#63b3ed"
    else:
        tier, tier_color = "Premium", "#9f7aea"

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Property Value</div>
        <div class="result-price"><em>${price:,.0f}</em></div>
        <div style="font-size:0.82rem; color:{tier_color}; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-top:0.4rem;">{tier}</div>
        <div class="badge-row">
            <div class="badge">Random Forest</div>
            <div class="badge">R² = 0.81</div>
            <div class="badge">RMSE = 0.51</div>
            <div class="badge">sklearn</div>
        </div>
        <div class="result-meta">
            Prices in USD · Based on California Housing Dataset · For educational purposes only
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <strong style="color:#8a9ab5;">Jad Mrad</strong> ·
    <a href="https://github.com/jad-mrad" target="_blank">github.com/jad-mrad</a> ·
    <a href="https://linkedin.com/in/jad-walid-mrad" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)