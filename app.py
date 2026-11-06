import pickle
import streamlit as st
import pandas as pd
import time

st.set_page_config("CardioSynth ICU Momnitor", layout="wide")
st.title("🫀 CardioSynth | Live ICU Deterioration Monitor")
st.markdown("### Real-time Predictive Clinical Early warning System")

@st.cache_resource
def load_model():
    with open("cardiosynth_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

@st.cache_data
def load_stream_data():
    return pd.read_csv("icu_mortality_prediction.csv")

df = load_stream_data()

drop_cols = ["mortality_label", "patient_id", "id", "ID"]
features_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

st.sidebar.header("🕹️ Simulation Controls")
start_stream = st.sidebar.button("▶️ Start Live Patient Stream", type="primary")
speed = st.sidebar.slider("Streaming Speed (seconds per update)", 0.5, 3.0, 1.0)