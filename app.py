import pickle
import streamlit as st
import pandas as pd
import time

st.set_page_config("CardioSynth ICU Momnitor", layout="wide")
st.title("🫀 CardioSynth | Live ICU Deterioration Monitor")
st.markdown("### Real-time Predictive Clinical Early warning System")

if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "clinical_log" not in st.session_state:
    st.session_state.clinical_log = []
    
@st.cache_resource
def load_model():
    with open("cardiosynth_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

@st.cache_data
def load_stream_data():
    return pd.read_csv("icu_mortality_prediction.csv")

df = load_stream_data()

drop_cols = ["mortality_label", "lmortality_label", "patient_id", "id", "ID"]
features_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

st.sidebar.header("🕹️ Simulation Controls")
speed = st.sidebar.slider("Streaming Speed (seconds per update)", 0.5, 3.0, 1.0)
if st.session_state.is_running:
    if st.sidebar.button("Start Live Patient Stream", type="primary"):
        st.session_state.is_running = True
        st.rerun()
else:
    if st.sidebar.button("Pause Live Stream", type="secondary"):
        st.session_state.is_running = False
        st.rerun()
monitor_column, log_column = st.columns([2.2, 1])
with monitor_column:
    dashboard_container = st.empty()

if st.session_state.is_running:
    processed_features = pd.get_dummies(features_df, columns=['gender', 'admission_type'], drop_first=True)
    expected_cols = [
        'age', 'comorbidity_score', 'heart_rate_mean', 'heart_rate_std', 'heart_rate_max', 'heart_rate_min', 
        'systolic_bp_mean', 'systolic_bp_std', 'respiratory_rate_mean', 'spo2_mean', 'temperature_mean', 
        'glucose_mean', 'lactate_mean', 'urine_output_total', 'ventilation_required', 'vasopressor_used', 
        'length_of_stay_days', 'apache_score', 'sofa_score', 'sepsis_flag', 'gender_Male', 
        'admission_type_Emergency', 'admission_type_Urgent'
    ]
    processed_features = processed_features.reindex(columns=expected_cols, fill_value=0)
    
    while st.session_state.current_index < len(features_df) and st.session_state.is_running:
        idx = st.session_state.current_index
        row = features_df.iloc[idx]
        single_patient_data = processed_features.iloc[[idx]]
        risk_list = model.predict_proba(single_patient_data)
        percent = float(risk_list[0][1]) * 100

        if percent >= 65:
            log_msg = f"**Bed #{patient_id}**: CRITICAL Alert ({percent:.1f}%). Rapid response paged."
            if log_msg not in st.session_state.clinical_log:
                st.session_state.clinical_log.insert(0, log_msg)
        elif percent >= 40:
            log_msg = f"**Bed #{patient_id}**: Elevated Risk ({percent:.1f}%). Nurse flagged."
            if log_msg not in st.session_state.clinical_log:
                st.session_state.clinical_log.insert(0, log_msg)

        with dashboard_container.container():  
            st.markdown(f"Analyzing Patient Unit ID: #{idx + 1042}")

            col1, col2, col3, col4 = st.columns(4)

            hr_col = [c for c in features_df.columns if 'heart' in c.lower()]
            bp_col = [c for c in features_df.columns if 'bp' in c.lower() or 'sys' in c.lower() or 'blood' in c.lower()]
            temp_col = [c for c in features_df.columns if 'temp' in c.lower()]
            spo2_col = [c for c in features_df.columns if 'spo2' in c.lower() or 'ox' in c.lower()]
            
            with col1:
                val = f"{row[hr_col[0]]:.1f} bpm" if hr_col else "85.0 bpm"
                st.metric(label="Heart Rate", value=val)
            with col2:
                val = f"{row[bp_col[0]]:.1f} mmHg" if bp_col else "118 mmHg"
                st.metric(label="Systolic Blood Pressure", value=val)
            with col3:
                val = f"{row[temp_col[0]]:.1f} °C" if temp_col else "36.9 °C"
                st.metric(label="Temperature", value=val)
            with col4:
                val = f"{row[spo2_col[0]]:.1f} %" if spo2_col else "98.2%"
                st.metric(label="SpO2 Oxygen", value=val) 

            st.markdown(f"CardioSynth Risk Probability: **{percent:.1f}%** chance of deterioration")

            if percent >= 65:
                st.error(f"CRITICAL WARNING: High risk of clinical deterioration ({percent:.1f}%). High medical priority.")
            elif percent >= 40:
                st.warning(f"HIGH RISK: Modest clinical instability found ({percent:.1f}%). Increase nursing checks.")
            else: 
                st.success(f"STABLE: Patient metrics are within healthy ranges ({percent:.1f}%).")
            
        st.session_state.current_index += 1
        if not st.session_state.is_running:
            break
        time.sleep(speed)
        
    if st.session_state.current_index >= len(features_df):
        st.session_state.is_running = False
        st.session_state.current_index = 0
        st.rerun()
else:
    if st.session_state.current_index > 0:
        idx = st.session_state.current_index - 1
        row = features_df.iloc[idx]
        single_patient_data = processed_features.iloc[[idx]]
        risk_list = model.predict_proba(single_patient_data)
        percent = float(risk_list[0][1]) * 100
        
        with dashboard_container.container():
            st.markdown(f"## MONITOR PAUSED // Patient Unit ID: #{idx + 1042}")
            col1, col2, col3, col4 = st.columns(4)
            hr_col = [c for c in features_df.columns if 'heart' in c.lower()]
            bp_col = [c for c in features_df.columns if 'bp' in c.lower() or 'sys' in c.lower() or 'blood' in c.lower()]
            temp_col = [c for c in features_df.columns if 'temp' in c.lower()]
            spo2_col = [c for c in features_df.columns if 'spo2' in c.lower() or 'ox' in c.lower()]
            with col1: st.metric(label="Heart Rate", value=f"{row[hr_col[0]]:.1f} bpm" if hr_col else "85.0 bpm")
            with col2: st.metric(label="Systolic Blood Pressure", value=f"{row[bp_col[0]]:.1f} mmHg" if bp_col else "118 mmHg")
            with col3: st.metric(label="Temperature", value=f"{row[temp_col[0]]:.1f} °C" if temp_col else "36.9 °C")
            with col4: st.metric(label="SpO2 Oxygen", value=f"{row[spo2_col[0]]:.1f} %" if spo2_col else "98.2%")
            st.markdown("---")
            st.markdown(f"CardioSynth Risk Probability: **{percent:.1f}%** chance of deterioration")
            if percent >= 65: st.error(f"CRITICAL WARNING: High risk of clinical deterioration ({percent:.1f}%). High medical priority.")
            elif percent >= 40: st.warning(f"HIGH RISK: Modest clinical instability found ({percent:.1f}%). Increase nursing checks.")
            else: st.success(f"STABLE: Patient metrics are within healthy ranges ({percent:.1f}%).")
    else:
        with dashboard_container.container():
            st.info('Click the "Start Live Patient Stream" in the left sidebar to begin processing clinical data.')
with log_column:
    st.markdown("### Nurse Audit Log")
    st.markdown("*Automated actions logged by model risk tracking:*")
    st.markdown("---")
    if st.session_state.clinical_log:
        for entry in st.session_state.clinical_log:
            st.markdown(entry)
    else:
        st.caption("Awaiting clinical alerts... Critical actions will appear here.")   