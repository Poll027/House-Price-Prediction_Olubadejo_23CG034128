import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction System")

# ---- MODEL LOADING (ROBUST) ----
MODEL_PATH = Path(__file__).parent / "model" / "house_price_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Model could not be loaded. Please check deployment files.")
    st.stop()

# ---- USER INPUT ----
st.subheader("Enter House Features")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality", 1, 10, 5)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 5000, 800)

with col2:
    garage_cars = st.selectbox("Garage Capacity", [0, 1, 2, 3, 4])
    year_built = st.number_input("Year Built", 1870, 2024, 2000)
    neighborhood = st.selectbox(
        "Neighborhood",
        ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", "Other"]
    )

input_df = pd.DataFrame([{
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "TotalBsmtSF": total_bsmt_sf,
    "GarageCars": garage_cars,
    "YearBuilt": year_built,
    "Neighborhood": neighborhood
}])

st.write("### Input Summary")
st.dataframe(input_df)

# ---- PREDICTION ----
if st.button("Predict House Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated House Price: ₦{prediction:,.2f}")
    except Exception:
        st.error("Prediction failed. Please review input values.")
