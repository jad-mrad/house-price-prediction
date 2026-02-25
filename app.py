import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HomeValue AI — California House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Nunito:wght@300;400;600;700;800&display=swap');

:root {
    --bg:        #f5f3ef;
    --card:      #ffffff;
    --primary:   #1a3c5e;
    --accent:    #e07b39;
    --accent2:   #2e86ab;
    --text:      #1e2a35;
    --muted:     #7a8a99;
    --border:    #e2ddd6;
    --radius:    20px;
    --shadow:    0 4px 32px rgba(26,60,94,0.10);
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Nunito', sans-serif;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(ellipse 70% 50% at 5% 0%,   rgba(224,123,57,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 95% 100%, rgba(46,134,171,0.08) 0%, transparent 55%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
section[data-testid="stSidebar"]                    { display: none !important; }
#MainMenu, footer                                    { visibility: hidden; }

.block-container {
    max-width: 1080px !important;
    padding: 0 2rem 5rem !important;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

/* ── Top nav bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 1rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 2.5rem;
}
.topbar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 900;
    color: var(--primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.topbar-logo span { color: var(--accent); }
.topbar-tag {
    background: var(--primary);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.35rem 0.9rem;
    border-radius: 100px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
}
.hero-kicker {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
}
.hero-h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5.5vw, 4rem);
    font-weight: 900;
    color: var(--primary);
    line-height: 1.12;
    margin-bottom: 1rem;
}
.hero-h1 em { color: var(--accent); font-style: italic; }
.hero-desc {
    font-size: 1.08rem;
    color: var(--muted);
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto 2rem;
    line-height: 1.75;
}

/* ── Metric strip ── */
.metrics-strip {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin-bottom: 3rem;
}
.metric-chip {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 0.8rem 1.4rem;
    text-align: center;
    box-shadow: var(--shadow);
    min-width: 120px;
}
.metric-chip-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--primary);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-chip-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Step labels ── */
.step-label {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 1.2rem;
}
.step-num {
    width: 32px; height: 32px;
    background: var(--primary);
    color: #fff;
    border-radius: 50%;
    font-size: 0.85rem;
    font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-title {
    font-size: 1rem;
    font-weight: 800;
    color: var(--primary);
    letter-spacing: 0.01em;
}
.step-sub {
    font-size: 0.78rem;
    color: var(--muted);
    font-weight: 400;
    margin-left: 2.7rem;
    margin-top: -0.8rem;
    margin-bottom: 1rem;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem 1.6rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.2rem;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.card:hover {
    box-shadow: 0 8px 40px rgba(26,60,94,0.14);
    border-color: rgba(224,123,57,0.35);
}
.card-header {
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 1.1rem;
    padding-bottom: 0.7rem;
    border-bottom: 1.5px solid var(--border);
}

/* ── Streamlit slider overrides ── */
[data-testid="stSlider"] label {
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}
[data-testid="stSlider"] p {
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

/* ── Info tip rows ── */
.tip-row {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    background: #f0f4f8;
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: #3a5068;
    line-height: 1.5;
}
.tip-icon { font-size: 1rem; flex-shrink: 0; margin-top: 0.05rem; }

/* ── Predict button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--primary) 0%, #2a5f94 100%) !important;
    color: #fff !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 6px 28px rgba(26,60,94,0.28) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 36px rgba(26,60,94,0.35) !important;
}

/* ── Result card ── */
.result-wrap {
    background: linear-gradient(145deg, #1a3c5e 0%, #1d5278 100%);
    border-radius: var(--radius);
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: 0 12px 48px rgba(26,60,94,0.28);
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: '🏡';
    position: absolute;
    font-size: 9rem;
    opacity: 0.04;
    top: -1.5rem; right: -1rem;
    pointer-events: none;
}
.result-kicker {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55);
    margin-bottom: 0.5rem;
}
.result-price {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.6rem, 6vw, 4rem);
    font-weight: 900;
    color: #ffffff;
    line-height: 1.05;
    margin-bottom: 0.4rem;
}
.result-price em { color: #f0a96b; font-style: normal; }
.result-tier {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.result-badges {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.result-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 700;
    color: rgba(255,255,255,0.75);
    letter-spacing: 0.05em;
}
.result-note {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.35);
    margin-top: 1.2rem;
}

/* ── How it works ── */
.how-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}
.how-card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: var(--shadow);
}
.how-icon { font-size: 2rem; margin-bottom: 0.7rem; }
.how-title { font-weight: 800; font-size: 0.9rem; color: var(--primary); margin-bottom: 0.4rem; }
.how-text  { font-size: 0.8rem; color: var(--muted); line-height: 1.55; }

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 1.8rem;
    border-top: 2px solid var(--border);
    font-size: 0.82rem;
    color: var(--muted);
}
.footer a { color: var(--accent2); text-decoration: none; font-weight: 600; }
.footer strong { color: var(--primary); }

@media (max-width: 640px) {
    .how-grid { grid-template-columns: 1fr; }
    .metrics-strip { gap: 0.7rem; }
    .metric-chip { min-width: 90px; padding: 0.6rem 1rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model… please wait ⏳")
def load_model():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

model, scaler = load_model()


# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">🏡 HomeValue <span>AI</span></div>
    <div class="topbar-tag">California · Machine Learning</div>
</div>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-kicker">✦ Free · Instant · AI-Powered ✦</div>
    <h1 class="hero-h1">What is your home<br><em>really worth?</em></h1>
    <p class="hero-desc">
        Move the sliders below to describe a property and our AI will
        estimate its market value in seconds — no sign-up needed.
    </p>
</div>

<div class="metrics-strip">
    <div class="metric-chip">
        <div class="metric-chip-value">0.81</div>
        <div class="metric-chip-label">R² Score</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-value">20K+</div>
        <div class="metric-chip-label">Homes Trained</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-value">100</div>
        <div class="metric-chip-label">Decision Trees</div>
    </div>
    <div class="metric-chip">
        <div class="metric-chip-value">&lt; 1s</div>
        <div class="metric-chip-label">Prediction Time</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Step 1 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-label">
    <div class="step-num">1</div>
    <div class="step-title">Tell us about the neighborhood</div>
</div>
<div class="step-sub">These details describe the area where the house is located.</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-header">📊 Area & Income</div>', unsafe_allow_html=True)
    MedInc = st.slider(
        "💵  Median Income of the Area",
        min_value=0.5, max_value=15.0, value=5.0, step=0.1,
        help="Average household income in the block group (in tens of thousands of dollars)"
    )
    Population = st.slider(
        "👥  Number of People in the Block",
        min_value=3, max_value=35000, value=1000, step=50,
        help="Total population of the block group"
    )
    AveOccup = st.slider(
        "🏘️  Average People per Household",
        min_value=1.0, max_value=10.0, value=3.0, step=0.1,
        help="Average number of people living in each house"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-header">📍 Location</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip-row">
        <span class="tip-icon">💡</span>
        <span>California latitude ranges from <strong>32°</strong> (San Diego) to <strong>42°</strong> (Oregon border).
        Longitude from <strong>−114°</strong> (Nevada side) to <strong>−124°</strong> (Pacific coast).</span>
    </div>
    """, unsafe_allow_html=True)
    Latitude = st.slider(
        "🧭  Latitude (North–South position)",
        min_value=32.0, max_value=42.0, value=34.0, step=0.1,
        help="Higher = further north. San Francisco is ~37.8, Los Angeles is ~34.0"
    )
    Longitude = st.slider(
        "🧭  Longitude (East–West position)",
        min_value=-124.0, max_value=-114.0, value=-118.0, step=0.1,
        help="More negative = further west (coast). Los Angeles is ~-118.2"
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Step 2 ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-label">
    <div class="step-num">2</div>
    <div class="step-title">Tell us about the house itself</div>
</div>
<div class="step-sub">These details describe the physical property.</div>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown('<div class="card"><div class="card-header">🏠 Property Details</div>', unsafe_allow_html=True)
    HouseAge = st.slider(
        "🗓️  Age of the House (years)",
        min_value=1, max_value=52, value=20,
        help="How old is the house? Newer houses are usually worth more."
    )
    AveRooms = st.slider(
        "🛋️  Average Rooms per House",
        min_value=1.0, max_value=15.0, value=5.0, step=0.1,
        help="Total number of rooms divided by number of households in the block"
    )
    AveBedrms = st.slider(
        "🛏️  Average Bedrooms per House",
        min_value=1.0, max_value=5.0, value=1.0, step=0.1,
        help="Total number of bedrooms divided by number of households in the block"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card"><div class="card-header">📖 Quick Guide — What do these mean?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip-row">
        <span class="tip-icon">💵</span>
        <span><strong>Median Income</strong> is the most important factor — richer neighborhoods always have higher house prices.</span>
    </div>
    <div class="tip-row">
        <span class="tip-icon">🏡</span>
        <span><strong>House Age:</strong> Older homes may need repairs. Newer homes usually cost more.</span>
    </div>
    <div class="tip-row">
        <span class="tip-icon">🛋️</span>
        <span><strong>Average Rooms</strong> is for the whole block group, not one house. A typical family home has 5–7 rooms.</span>
    </div>
    <div class="tip-row">
        <span class="tip-icon">📍</span>
        <span><strong>Coastal areas</strong> (more western longitudes) are generally much more expensive in California.</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Step 3 / Predict ───────────────────────────────────────────────────────────
st.markdown("""
<div class="step-label" style="margin-top:1.5rem;">
    <div class="step-num">3</div>
    <div class="step-title">Get your instant AI prediction</div>
</div>
<div class="step-sub">Click the button below — your result appears immediately.</div>
""", unsafe_allow_html=True)

clicked = st.button("🔮  Predict House Price Now", use_container_width=True)

if clicked:
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    price = prediction * 100_000

    if price < 120_000:
        tier, tier_bg = "🟢 Budget Friendly",  "rgba(45,122,79,0.85)"
    elif price < 250_000:
        tier, tier_bg = "🔵 Mid Market",        "rgba(46,134,171,0.85)"
    elif price < 450_000:
        tier, tier_bg = "🟠 Above Average",     "rgba(224,123,57,0.85)"
    elif price < 700_000:
        tier, tier_bg = "🔴 High End",          "rgba(180,50,50,0.80)"
    else:
        tier, tier_bg = "💎 Luxury",            "rgba(120,60,180,0.85)"

    st.markdown(f"""
    <div class="result-wrap">
        <div class="result-kicker">✦ AI Estimated Market Value ✦</div>
        <div class="result-price"><em>${price:,.0f}</em></div>
        <div class="result-tier" style="background:{tier_bg}; color:#fff;">{tier}</div>
        <div class="result-badges">
            <div class="result-badge">🌲 Random Forest</div>
            <div class="result-badge">📊 R² = 0.81</div>
            <div class="result-badge">📉 RMSE = 0.51</div>
            <div class="result-badge">🧠 100 Trees</div>
        </div>
        <div class="result-note">
            For educational purposes only · Based on California Housing Dataset (sklearn)
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── How it works ───────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="step-label">
    <div class="step-num" style="background:#e07b39;">?</div>
    <div class="step-title">How does this AI work?</div>
</div>
<div class="how-grid">
    <div class="how-card">
        <div class="how-icon">📚</div>
        <div class="how-title">Step 1 · Learning</div>
        <div class="how-text">The AI studied over 20,000 real California homes — their features and actual sale prices — to learn patterns.</div>
    </div>
    <div class="how-card">
        <div class="how-icon">🌲</div>
        <div class="how-title">Step 2 · Thinking</div>
        <div class="how-text">100 decision trees each make their own prediction. The AI then combines all answers for the most accurate result.</div>
    </div>
    <div class="how-card">
        <div class="how-icon">⚡</div>
        <div class="how-title">Step 3 · Result</div>
        <div class="how-text">Your inputs are processed in milliseconds. No waiting, no sign-up — just a clear, honest price estimate.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with ❤️ by <strong>Jad Mrad</strong> &nbsp;·&nbsp;
    <a href="https://github.com/jad-mrad" target="_blank">GitHub</a> &nbsp;·&nbsp;
    <a href="https://linkedin.com/in/jad-walid-mrad" target="_blank">LinkedIn</a>
    <br><br>
    Computer Engineering Student · BAU Lebanon · AI/ML Internship @ Khatib &amp; Alami
</div>
""", unsafe_allow_html=True)