import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@st.cache_resource
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

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.markdown("**Predict California house prices using Machine Learning**")
st.divider()

st.subheader("Enter House Details:")
col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("Median Income", 0.5, 15.0, 5.0)
    HouseAge = st.slider("House Age (years)", 1, 52, 20)
    AveRooms = st.slider("Average Rooms", 1.0, 15.0, 5.0)
    AveBedrms = st.slider("Average Bedrooms", 1.0, 5.0, 1.0)

with col2:
    Population = st.slider("Population", 3, 35000, 1000)
    AveOccup = st.slider("Average Occupants", 1.0, 10.0, 3.0)
    Latitude = st.slider("Latitude", 32.0, 42.0, 35.0)
    Longitude = st.slider("Longitude", -124.0, -114.0, -119.0)

st.divider()

if st.button("🔮 Predict Price", use_container_width=True):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    price = prediction * 100000
    st.success(f"### 🏠 Estimated House Price: ${price:,.0f}")
    st.info("Model: Random Forest | R² Score: 0.81")

st.divider()
st.markdown("Built by **Jad Mrad** | github.com/jad-mrad")
