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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,700;1,9..144,400&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    /* Warm cream & sage — gentle on the eyes */
    --bg:           #FAF8F4;
    --card:         #FFFFFF;
    --primary:      #5B7B6A;
    --primary-deep: #3D5A4C;
    --primary-soft: #8BA89A;
    --accent:       #C4956A;
    --accent-soft:  #D4AD87;
    --text:         #2C2C2C;
    --text-soft:    #787470;
    --text-muted:   #A8A29E;
    --border:       #EAE6DF;
    --border-warm:  #DDD8CF;
    --input-bg:     #F7F5F0;
    --glow:         rgba(91,123,106,0.08);
    --radius:       22px;
    --radius-sm:    14px;
    --shadow-soft:  0 2px 16px rgba(44,44,44,0.04);
    --shadow-card:  0 4px 24px rgba(44,44,44,0.05);
    --shadow-hover: 0 8px 32px rgba(44,44,44,0.08);
    --transition:   all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Outfit', -apple-system, sans-serif !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 0% 0%, rgba(196,149,106,0.05) 0%, transparent 50%),
        radial-gradient(ellipse 60% 50% at 100% 100%, rgba(91,123,106,0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"],
[data-testid="stToolbar"]       { display: none !important; }
section[data-testid="stSidebar"]{ display: none !important; }
#MainMenu, footer               { visibility: hidden; }

.block-container {
    max-width: 920px !important;
    padding: 0 1.5rem 5rem !important;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

/* ── Navbar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.3rem 0;
    margin-bottom: 1rem;
}
.nav-left {
    display: flex;
    align-items: center;
    gap: 11px;
}
.nav-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-soft) 100%);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    color: white;
    box-shadow: 0 2px 10px rgba(91,123,106,0.2);
}
.nav-name {
    font-family: 'Fraunces', serif;
    font-size: 1.3rem;
    font-weight: 500;
    color: var(--primary-deep);
    letter-spacing: -0.4px;
}
.nav-right {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--primary);
    background: var(--glow);
    padding: 6px 14px;
    border-radius: 100px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.nav-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--border) 20%, var(--border) 80%, transparent 100%);
    margin-bottom: 2rem;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Fraunces', serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 300;
    color: var(--primary-deep);
    line-height: 1.2;
    margin-bottom: 1rem;
    letter-spacing: -0.5px;
}
.hero h1 em {
    font-weight: 500;
    font-style: italic;
    color: var(--primary);
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-soft);
    font-weight: 400;
    max-width: 440px;
    margin: 0 auto;
    line-height: 1.8;
}

/* Trust pills */
.trust-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin: 2rem 0 2.5rem;
}
.trust-pill {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-soft);
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 7px 16px;
    box-shadow: var(--shadow-soft);
}
.trust-pill span { font-size: 0.85rem; }

/* ── Section headers ── */
.sec-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.5rem;
}
.sec-dot {
    width: 28px; height: 28px;
    background: var(--primary);
    color: #fff;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(91,123,106,0.2);
}
.sec-title {
    font-family: 'Fraunces', serif;
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--primary-deep);
}
.sec-desc {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-left: 40px;
    margin-bottom: 1rem;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.5rem;
    box-shadow: var(--shadow-card);
    margin-bottom: 1rem;
    transition: var(--transition);
}
.card:hover {
    box-shadow: var(--shadow-hover);
    border-color: var(--border-warm);
}
.card-tag {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--primary-soft);
    margin-bottom: 1rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--border);
}

/* ── Tips ── */
.tip {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    background: linear-gradient(135deg, #F5F1EA 0%, #F0EDE5 100%);
    border-left: 3px solid var(--accent-soft);
    border-radius: 10px;
    padding: 11px 14px;
    margin-bottom: 8px;
    font-size: 0.8rem;
    color: #5C554E;
    line-height: 1.6;
}
.tip strong { color: #4A4440; }
.tip-i { font-size: 0.9rem; flex-shrink: 0; margin-top: 1px; }

/* ── Slider overrides ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] label p {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    color: var(--primary-deep) !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-deep) 100%) !important;
    color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 1rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(61,90,76,0.22) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(61,90,76,0.3) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Result ── */
.result-wrap {
    background: linear-gradient(150deg, #5B7B6A 0%, #3D5A4C 60%, #2F473C 100%);
    border-radius: var(--radius);
    padding: 3rem 2rem;
    text-align: center;
    box-shadow:
        0 16px 48px rgba(61,90,76,0.22),
        inset 0 1px 0 rgba(255,255,255,0.08);
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: '';
    position: absolute;
    top: -50%; left: 20%;
    width: 60%; height: 120%;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.result-wrap::after {
    content: '';
    position: absolute;
    bottom: -30%; right: -10%;
    width: 50%; height: 80%;
    background: radial-gradient(circle, rgba(196,149,106,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.res-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45);
    margin-bottom: 0.7rem;
    position: relative;
}
.res-price {
    font-family: 'Fraunces', serif;
    font-size: clamp(2.4rem, 7vw, 3.8rem);
    font-weight: 300;
    color: #fff;
    line-height: 1.05;
    margin-bottom: 0.6rem;
    position: relative;
}
.res-price em {
    font-weight: 500;
    font-style: normal;
    color: #C8E0D2;
}
.res-tier {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 5px 18px;
    border-radius: 100px;
    color: #fff;
    margin-bottom: 1.2rem;
    position: relative;
}
.res-pills {
    display: flex;
    gap: 7px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 0.8rem;
    position: relative;
}
.res-pill {
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    padding: 6px 13px;
    font-size: 0.72rem;
    font-weight: 500;
    color: rgba(255,255,255,0.65);
}
.res-note {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.28);
    margin-top: 1.6rem;
    line-height: 1.6;
    position: relative;
}

/* ── How it works ── */
.how-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-top: 0.8rem;
}
.how-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem 1.2rem;
    text-align: center;
    box-shadow: var(--shadow-soft);
    transition: var(--transition);
}
.how-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-hover);
}
.how-ic { font-size: 1.7rem; margin-bottom: 0.7rem; }
.how-h {
    font-family: 'Fraunces', serif;
    font-weight: 500;
    font-size: 0.9rem;
    color: var(--primary-deep);
    margin-bottom: 0.4rem;
}
.how-p {
    font-size: 0.78rem;
    color: var(--text-soft);
    line-height: 1.6;
}

/* ── Footer ── */
.site-footer {
    text-align: center;
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.8;
}
.site-footer strong {
    color: var(--text-soft);
    font-weight: 600;
}
.site-footer a {
    color: var(--primary);
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s;
}
.site-footer a:hover {
    color: var(--accent);
}
.footer-role {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .block-container { padding: 0 1rem 4rem !important; }
    .hero { padding: 2rem 0.5rem 1rem; }
    .how-grid { grid-template-columns: 1fr; gap: 10px; }
    .result-wrap { padding: 2.2rem 1.3rem; }
    .card { padding: 1.3rem 1.1rem; }
    .trust-row { gap: 6px; }
}
@media (max-width: 480px) {
    .navbar { flex-direction: column; gap: 8px; align-items: flex-start; }
    .trust-row { flex-direction: column; align-items: center; }
    .trust-pill { width: 100%; justify-content: center; }
}
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
    <div class="nav-left">
        <div class="nav-icon">⌂</div>
        <span class="nav-name">HomeValue</span>
    </div>
    <div class="nav-right">✦ AI Estimator</div>
</div>
<div class="nav-divider"></div>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">California Property Estimator</div>
    <h1>Discover What Your<br>Home Is <em>Really Worth</em></h1>
    <p class="hero-sub">
        Describe a property using the sliders below and get an
        instant AI estimate — completely free, no sign-up.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="trust-row">
    <div class="trust-pill"><span>⚡</span> Instant results</div>
    <div class="trust-pill"><span>🔒</span> Private & secure</div>
    <div class="trust-pill"><span>🆓</span> Always free</div>
    <div class="trust-pill"><span>📱</span> Mobile friendly</div>
</div>
""", unsafe_allow_html=True)


# ── Step 1 — Neighborhood ─────────────────────────────────────────────────────
st.markdown("""
<div class="sec-header">
    <div class="sec-dot">1</div>
    <div class="sec-title">About the Neighborhood</div>
</div>
<div class="sec-desc">Describe the area where the property is located.</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-tag">Area & Demographics</div>', unsafe_allow_html=True)
    MedInc = st.slider(
        "💵  Median Household Income",
        min_value=0.5, max_value=15.0, value=5.0, step=0.1,
        help="Average income in the area (in $10,000s). For example, 5.0 means $50,000/year."
    )
    Population = st.slider(
        "👥  Local Population",
        min_value=3, max_value=35000, value=1000, step=50,
        help="Total number of people living in this neighborhood block."
    )
    AveOccup = st.slider(
        "🏘️  People per Household",
        min_value=1.0, max_value=10.0, value=3.0, step=0.1,
        help="Average number of people living in each nearby home."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-tag">Location in California</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip">
        <span class="tip-i">📍</span>
        <span><strong>Quick reference:</strong> San Francisco ≈ 37.8°N, 122.4°W · Los Angeles ≈ 34.0°N, 118.2°W · San Diego ≈ 32.7°N, 117.2°W</span>
    </div>
    """, unsafe_allow_html=True)
    Latitude = st.slider(
        "🧭  Latitude (North — South)",
        min_value=32.0, max_value=42.0, value=34.0, step=0.1,
        help="Higher values = further north. San Francisco is ~37.8, San Diego is ~32.7"
    )
    Longitude = st.slider(
        "🧭  Longitude (East — West)",
        min_value=-124.0, max_value=-114.0, value=-118.0, step=0.1,
        help="More negative = closer to the Pacific coast. LA is about -118.2"
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Step 2 — Property ─────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-header">
    <div class="sec-dot">2</div>
    <div class="sec-title">About the Property</div>
</div>
<div class="sec-desc">Details about the house — age, size, and rooms.</div>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown('<div class="card"><div class="card-tag">Property Details</div>', unsafe_allow_html=True)
    HouseAge = st.slider(
        "🗓️  Age of the House (years)",
        min_value=1, max_value=52, value=20,
        help="How old the property is. Newer homes are generally valued higher."
    )
    AveRooms = st.slider(
        "🛋️  Average Rooms per House",
        min_value=1.0, max_value=15.0, value=5.0, step=0.1,
        help="Typical number of rooms in homes in this area. A family home usually has 5–7."
    )
    AveBedrms = st.slider(
        "🛏️  Average Bedrooms per House",
        min_value=1.0, max_value=5.0, value=1.0, step=0.1,
        help="Typical number of bedrooms in homes in this area."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card"><div class="card-tag">Helpful Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip">
        <span class="tip-i">💵</span>
        <span><strong>Income is key</strong> — wealthier neighborhoods almost always have higher property values.</span>
    </div>
    <div class="tip">
        <span class="tip-i">🏡</span>
        <span><strong>Age matters</strong> — newer homes tend to be worth more, while older ones may need renovation.</span>
    </div>
    <div class="tip">
        <span class="tip-i">🌊</span>
        <span><strong>Coast = premium</strong> — properties closer to the Pacific (more negative longitude) are typically pricier.</span>
    </div>
    <div class="tip">
        <span class="tip-i">🛋️</span>
        <span><strong>Room count</strong> — more rooms generally means a larger home and a higher price tag.</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Step 3 — Predict ──────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-header" style="margin-top:1.5rem;">
    <div class="sec-dot">3</div>
    <div class="sec-title">Get Your Estimate</div>
</div>
<div class="sec-desc">Click below — your result will appear instantly.</div>
""", unsafe_allow_html=True)

clicked = st.button("🏡  Estimate Property Value", use_container_width=True)

if clicked:
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    price = prediction * 100_000

    if price < 120_000:
        tier, tier_bg = "Budget Friendly",  "rgba(91,123,106,0.85)"
    elif price < 250_000:
        tier, tier_bg = "Mid Range",        "rgba(70,130,160,0.85)"
    elif price < 450_000:
        tier, tier_bg = "Above Average",    "rgba(196,149,106,0.9)"
    elif price < 700_000:
        tier, tier_bg = "Premium",          "rgba(175,95,65,0.85)"
    else:
        tier, tier_bg = "Luxury",           "rgba(135,85,160,0.85)"

    income_fmt = f"${MedInc * 10_000:,.0f}/yr"
    loc_fmt = f"{Latitude:.1f}°N, {abs(Longitude):.1f}°W"

    st.markdown(f"""
    <div class="result-wrap">
        <div class="res-eyebrow">Estimated Market Value</div>
        <div class="res-price"><em>${price:,.0f}</em></div>
        <div class="res-tier" style="background:{tier_bg};">{tier}</div>
        <div class="res-pills">
            <div class="res-pill">🗓️ {HouseAge} yrs old</div>
            <div class="res-pill">🛋️ {AveRooms:.1f} rooms</div>
            <div class="res-pill">💵 {income_fmt} area</div>
            <div class="res-pill">📍 {loc_fmt}</div>
        </div>
        <div class="res-note">
            This is an AI estimate for informational purposes only.<br>
            Actual values depend on market conditions, property specifics, and more.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── How it works ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
    <div class="sec-dot" style="background:var(--accent);">?</div>
    <div class="sec-title">How It Works</div>
</div>
<div class="how-grid">
    <div class="how-card">
        <div class="how-ic">📊</div>
        <div class="how-h">Learns from Data</div>
        <div class="how-p">Our AI studied real California housing data to understand what drives property prices.</div>
    </div>
    <div class="how-card">
        <div class="how-ic">🧠</div>
        <div class="how-h">Finds Patterns</div>
        <div class="how-p">It considers location, income, home age, and size to calculate the best possible estimate.</div>
    </div>
    <div class="how-card">
        <div class="how-ic">⚡</div>
        <div class="how-h">Instant Answer</div>
        <div class="how-p">No sign-up, no waiting. Just adjust the sliders and get a clear estimate in seconds.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
    Built with ❤️ by <strong>Jad Mrad</strong> &nbsp;·&nbsp;
    <a href="https://github.com/jad-mrad" target="_blank">GitHub</a> &nbsp;·&nbsp;
    <a href="https://linkedin.com/in/jad-walid-mrad" target="_blank">LinkedIn</a>
    <div class="footer-role">Computer Engineering Student · BAU Lebanon</div>
</div>
""", unsafe_allow_html=True)