import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Multi-Disease Predictor", page_icon="🩺", layout="wide")
st.title("🩺 Multiple Disease Prediction System")
st.caption("Educational demo only. Not a medical diagnosis. Consult a doctor.")

DISEASES = ["Diabetes", "Heart Disease", "Breast Cancer"]
disease = st.sidebar.selectbox("Select disease", DISEASES)


@st.cache_resource
def load(name):
    return joblib.load(f"models/{name.replace(' ', '_').lower()}.joblib")


art = load(disease)
model, features = art["model"], art["features"]

st.subheader(f"{disease} Prediction")
st.write(f"Model accuracy: **{art['accuracy']:.1%}** | ROC-AUC: **{art['auc']:.3f}**")

cols = st.columns(3)
values = {}
for i, f in enumerate(features):
    with cols[i % 3]:
        values[f] = st.number_input(
            f,
            min_value=float(art["min"][f]),
            max_value=float(art["max"][f]),
            value=float(art["mean"][f]),
        )

if st.button("Predict", type="primary"):
    df = pd.DataFrame([values])[features]
    risk = model.predict_proba(df)[0][1]
    if risk >= 0.5:
        st.error(f"⚠️ High risk of {disease}: {risk:.1%}")
    else:
        st.success(f"✅ Low risk of {disease}: {risk:.1%}")
    st.progress(float(risk))