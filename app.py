import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HomeValue — California Property Estimator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital,wght@0,400;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:         #F7F6F3;
    --card:       #FFFFFF;
    --primary:    #2D6A4F;
    --primary-d:  #1B4332;
    --accent:     #D4A373;
    --accent-warm:#E07B39;
    --text:       #1A1A1A;
    --text-sec:   #6B7280;
    --border:     #E5E2DB;
    --input-bg:   #F2F0EB;
    --glow:       rgba(45,106,79,0.10);
    --radius:     18px;
    --shadow:     0 2px 20px rgba(0,0,0,0.05);
    --shadow-md:  0 8px 32px rgba(0,0,0,0.07);
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
}

/* Subtle background texture */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 45% at 10% 0%, rgba(45,106,79,0.06) 0%, transparent 50%),
        radial-gradient(ellipse 50% 40% at 90% 100%, rgba(212,163,115,0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* Hide Streamlit chrome */
[data-testid="stHeader"],
[data-testid="stToolbar"]       { display: none !important; }
section[data-testid="stSidebar"]{ display: none !important; }
#MainMenu, footer               { visibility: hidden; }

.block-container {
    max-width: 960px !important;
    padding: 0 1.5rem 5rem !important;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

/* ══════════════════════════════════════════════════
   NAVBAR
   ══════════════════════════════════════════════════ */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0;
    border-bottom: 1.5px solid var(--border);
    margin-bottom: 2rem;
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-logo-box {
    width: 36px; height: 36px;
    background: var(--primary);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    color: white;
    line-height: 1;
}
.nav-logo-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: var(--primary-d);
    letter-spacing: -0.3px;
}
.nav-badge {
    font-size: 0.68rem;
    font-weight: 700;
    background: var(--glow);
    color: var(--primary);
    padding: 6px 14px;
    border-radius: 100px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

/* ══════════════════════════════════════════════════
   HERO
   ══════════════════════════════════════════════════ */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.8rem;
}
.hero-tag {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--primary);
    opacity: 0.7;
    margin-bottom: 0.8rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 400;
    color: var(--primary-d);
    line-height: 1.15;
    margin-bottom: 0.9rem;
    letter-spacing: -0.5px;
}
.hero h1 em {
    color: var(--primary);
    font-style: italic;
}
.hero-desc {
    font-size: 1.05rem;
    color: var(--text-sec);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.75;
}

/* ══════════════════════════════════════════════════
   TRUST STRIP (replaces metrics strip)
   ══════════════════════════════════════════════════ */
.trust-strip {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin: 1.5rem 0 2.5rem;
    padding: 0 1rem;
}
.trust-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: var(--text-sec);
    font-weight: 500;
}
.trust-item span {
    font-size: 1rem;
}

/* ══════════════════════════════════════════════════
   SECTION HEADERS
   ══════════════════════════════════════════════════ */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.6rem;
}
.section-num {
    width: 30px; height: 30px;
    background: var(--primary);
    color: #fff;
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: var(--primary-d);
}
.section-sub {
    font-size: 0.82rem;
    color: var(--text-sec);
    margin-left: 42px;
    margin-bottom: 1rem;
    line-height: 1.5;
}

/* ══════════════════════════════════════════════════
   CARDS
   ══════════════════════════════════════════════════ */
.card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.4rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
    transition: box-shadow 0.25s, border-color 0.25s;
}
.card:hover {
    box-shadow: var(--shadow-md);
    border-color: rgba(45,106,79,0.2);
}
.card-label {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--primary);
    opacity: 0.65;
    margin-bottom: 1rem;
    padding-bottom: 0.65rem;
    border-bottom: 1.5px solid var(--border);
}

/* ══════════════════════════════════════════════════
   HELP TIPS
   ══════════════════════════════════════════════════ */
.tip {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    background: #F0F7F4;
    border-left: 3px solid var(--primary);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.8rem;
    color: #3A5D4C;
    line-height: 1.55;
}
.tip-icon { font-size: 0.95rem; flex-shrink: 0; margin-top: 1px; }

/* ══════════════════════════════════════════════════
   STREAMLIT SLIDER OVERRIDES
   ══════════════════════════════════════════════════ */
[data-testid="stSlider"] label,
[data-testid="stSlider"] label p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}

/* Slider track */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    font-weight: 700 !important;
    color: var(--primary) !important;
}

/* ══════════════════════════════════════════════════
   PREDICT BUTTON
   ══════════════════════════════════════════════════ */
[data-testid="stButton"] > button {
    background: var(--primary) !important;
    color: #fff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.95rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(45,106,79,0.25) !important;
    transition: transform 0.15s, box-shadow 0.15s, background 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    background: var(--primary-d) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(45,106,79,0.32) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ══════════════════════════════════════════════════
   RESULT CARD
   ══════════════════════════════════════════════════ */
.result-card {
    background: linear-gradient(145deg, #2D6A4F 0%, #1B4332 100%);
    border-radius: var(--radius);
    padding: 2.8rem 2rem;
    text-align: center;
    box-shadow: 0 12px 44px rgba(27,67,50,0.25);
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -40%; left: -20%;
    width: 140%; height: 140%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 55%);
    pointer-events: none;
}
.result-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.5);
    margin-bottom: 0.6rem;
}
.result-price {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.6rem, 7vw, 4rem);
    color: #fff;
    line-height: 1.05;
    margin-bottom: 0.5rem;
}
.result-price em {
    color: #A7D9BC;
    font-style: normal;
}
.result-tier {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 16px;
    border-radius: 100px;
    color: #fff;
    margin-bottom: 1.2rem;
}
.result-summary {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.result-pill {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(255,255,255,0.7);
}
.result-footnote {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.3);
    margin-top: 1.5rem;
    line-height: 1.5;
}

/* ══════════════════════════════════════════════════
   HOW IT WORKS
   ══════════════════════════════════════════════════ */
.how-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-top: 0.8rem;
}
.how-item {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    box-shadow: var(--shadow);
    transition: transform 0.2s, box-shadow 0.2s;
}
.how-item:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.how-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
.how-title {
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--primary-d);
    margin-bottom: 0.35rem;
}
.how-text {
    font-size: 0.78rem;
    color: var(--text-sec);
    line-height: 1.55;
}

/* ══════════════════════════════════════════════════
   FOOTER
   ══════════════════════════════════════════════════ */
.site-footer {
    text-align: center;
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1.5px solid var(--border);
    font-size: 0.78rem;
    color: var(--text-sec);
}

/* ══════════════════════════════════════════════════
   RESPONSIVE
   ══════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .block-container {
        padding: 0 1rem 4rem !important;
    }
    .hero { padding: 2rem 0.5rem 1.2rem; }
    .trust-strip { gap: 1rem; }
    .how-grid { grid-template-columns: 1fr; gap: 10px; }
    .result-card { padding: 2rem 1.2rem; }
    .card { padding: 1.3rem 1.1rem; }
}

@media (max-width: 480px) {
    .navbar { flex-direction: column; gap: 8px; align-items: flex-start; }
    .trust-strip { flex-direction: column; align-items: center; gap: 6px; }
}

/* Reduce motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing the estimator… ⏳")
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


# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-logo">
        <div class="nav-logo-box">⌂</div>
        <span class="nav-logo-text">HomeValue</span>
    </div>
    <div class="nav-badge">✦ AI Powered</div>
</div>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">California Property Estimator</div>
    <h1>Estimate Your Home's<br><em>True Value</em></h1>
    <p class="hero-desc">
        Adjust the sliders to describe a property and get an instant
        AI-powered price estimate — no sign-up needed.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Trust strip (no technical stats) ──────────────────────────────────────────
st.markdown("""
<div class="trust-strip">
    <div class="trust-item"><span>⚡</span> Instant results</div>
    <div class="trust-item"><span>🔒</span> No data stored</div>
    <div class="trust-item"><span>🆓</span> Completely free</div>
    <div class="trust-item"><span>📱</span> Works on any device</div>
</div>
""", unsafe_allow_html=True)


# ── Step 1 — Neighborhood ─────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-num">1</div>
    <div class="section-title">About the Neighborhood</div>
</div>
<div class="section-sub">Describe the area where the property is located.</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-label">Area Demographics</div>', unsafe_allow_html=True)
    MedInc = st.slider(
        "💵  Median Household Income",
        min_value=0.5, max_value=15.0, value=5.0, step=0.1,
        help="Average income in the area (in $10,000s). E.g. 5.0 = $50,000/year"
    )
    Population = st.slider(
        "👥  Local Population",
        min_value=3, max_value=35000, value=1000, step=50,
        help="Number of people living in this neighborhood block"
    )
    AveOccup = st.slider(
        "🏘️  People per Household",
        min_value=1.0, max_value=10.0, value=3.0, step=0.1,
        help="Average number of people in each home nearby"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-label">Location in California</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip">
        <span class="tip-icon">📍</span>
        <span><strong>San Francisco</strong> ≈ 37.8°N, 122.4°W &nbsp;·&nbsp;
        <strong>Los Angeles</strong> ≈ 34.0°N, 118.2°W &nbsp;·&nbsp;
        <strong>San Diego</strong> ≈ 32.7°N, 117.2°W</span>
    </div>
    """, unsafe_allow_html=True)
    Latitude = st.slider(
        "🧭  Latitude (North — South)",
        min_value=32.0, max_value=42.0, value=34.0, step=0.1,
        help="Higher number = further north in California"
    )
    Longitude = st.slider(
        "🧭  Longitude (East — West)",
        min_value=-124.0, max_value=-114.0, value=-118.0, step=0.1,
        help="More negative = closer to the coast"
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Step 2 — Property ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-num">2</div>
    <div class="section-title">About the Property</div>
</div>
<div class="section-sub">Describe the house itself — age, size, and rooms.</div>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown('<div class="card"><div class="card-label">Property Details</div>', unsafe_allow_html=True)
    HouseAge = st.slider(
        "🗓️  Age of the House (years)",
        min_value=1, max_value=52, value=20,
        help="How old the house is — newer homes tend to be valued higher"
    )
    AveRooms = st.slider(
        "🛋️  Average Rooms per House",
        min_value=1.0, max_value=15.0, value=5.0, step=0.1,
        help="Typical number of rooms in homes in this area"
    )
    AveBedrms = st.slider(
        "🛏️  Average Bedrooms per House",
        min_value=1.0, max_value=5.0, value=1.0, step=0.1,
        help="Typical number of bedrooms in homes in this area"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card"><div class="card-label">Quick Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip">
        <span class="tip-icon">💵</span>
        <span><strong>Income matters most</strong> — neighborhoods with higher incomes almost always have higher home prices.</span>
    </div>
    <div class="tip">
        <span class="tip-icon">🏡</span>
        <span><strong>Newer homes</strong> typically have higher values, while older homes may cost less due to wear.</span>
    </div>
    <div class="tip">
        <span class="tip-icon">🌊</span>
        <span><strong>Coastal areas</strong> (further west / more negative longitude) are generally more expensive in California.</span>
    </div>
    <div class="tip">
        <span class="tip-icon">🛋️</span>
        <span><strong>More rooms</strong> usually means larger homes and higher prices — a typical family home has 5–7 rooms.</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Step 3 — Predict ──────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header" style="margin-top:1.5rem;">
    <div class="section-num">3</div>
    <div class="section-title">Get Your Estimate</div>
</div>
<div class="section-sub">Click the button — your result appears instantly.</div>
""", unsafe_allow_html=True)

clicked = st.button("🏡  Estimate Property Value", use_container_width=True)

if clicked:
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    price = prediction * 100_000

    # Price tier
    if price < 120_000:
        tier, tier_bg = "Budget Friendly",  "rgba(45,122,79,0.85)"
    elif price < 250_000:
        tier, tier_bg = "Mid Market",        "rgba(46,134,171,0.85)"
    elif price < 450_000:
        tier, tier_bg = "Above Average",     "rgba(212,163,115,0.9)"
    elif price < 700_000:
        tier, tier_bg = "Premium",           "rgba(180,80,50,0.85)"
    else:
        tier, tier_bg = "Luxury",            "rgba(120,60,180,0.85)"

    # Format summary pills
    income_display = f"${MedInc * 10_000:,.0f}/yr income"
    location_display = f"{Latitude:.1f}°N, {abs(Longitude):.1f}°W"

    st.markdown(f"""
    <div class="result-card">
        <div class="result-eyebrow">Estimated Market Value</div>
        <div class="result-price"><em>${price:,.0f}</em></div>
        <div class="result-tier" style="background:{tier_bg};">{tier}</div>
        <div class="result-summary">
            <div class="result-pill">🗓️ {HouseAge} yrs old</div>
            <div class="result-pill">🛋️ {AveRooms:.1f} rooms</div>
            <div class="result-pill">💵 {income_display}</div>
            <div class="result-pill">📍 {location_display}</div>
        </div>
        <div class="result-footnote">
            This is an AI-generated estimate for informational purposes only.<br>
            Actual property values may vary based on market conditions.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── How it works ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <div class="section-num" style="background:var(--accent-warm);">?</div>
    <div class="section-title">How It Works</div>
</div>
<div class="how-grid">
    <div class="how-item">
        <div class="how-icon">📊</div>
        <div class="how-title">Learns from Data</div>
        <div class="how-text">Our AI studied real California housing data to understand what drives property prices.</div>
    </div>
    <div class="how-item">
        <div class="how-icon">🧠</div>
        <div class="how-title">Analyzes Patterns</div>
        <div class="how-text">It considers location, income, home age, and size to find the best estimate for you.</div>
    </div>
    <div class="how-item">
        <div class="how-icon">⚡</div>
        <div class="how-title">Instant Result</div>
        <div class="how-text">No waiting, no sign-up. Just adjust the sliders and get a clear price estimate in seconds.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
    © 2025 HomeValue — California Property Estimator<br>
    AI-powered estimates for informational purposes only.
</div>
""", unsafe_allow_html=True)