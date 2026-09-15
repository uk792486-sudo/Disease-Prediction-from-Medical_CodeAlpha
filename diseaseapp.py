import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Medical Diagnostics Portal",
    page_icon="🏥",
    layout="centered"
)

# Custom CSS for a professional HTML-like card layout & styling
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        background-color: #0284c7;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0369a1;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

try:
    model, scaler, feature_names = load_artifacts()
except Exception as e:
    st.error("Model files not found! Please run 'train_model.py' first to generate them.")
    st.stop()

# Header section
st.markdown("<h1 style='text-align: center; color: #0f172a;'>🏥 AI Medical Diagnostic Portal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Advanced Machine Learning Web App for Disease Risk Prediction (CodeAlpha Internship Task 4)</p>", unsafe_allow_html=True)
st.write("---")

st.markdown("### Enter Patient Biomarkers / Clinical Metrics")
st.markdown("Provide the measurements below to get real-time diagnostic risk assessment.")

# Form inputs arranged in columns
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    input_vals = []
    defaults = [14.12, 19.29, 91.97, 654.9, 0.096, 0.104, 0.088, 0.048, 0.181, 0.062]
    
    for i, feature in enumerate(feature_names):
        label = feature.title()
        if i % 2 == 0:
            with col1:
                val = st.number_input(label, value=float(defaults[i]), format="%.4f")
                input_vals.append(val)
        else:
            with col2:
                val = st.number_input(label, value=float(defaults[i]), format="%.4f")
                input_vals.append(val)
                
    submitted = st.form_submit_button("Run Diagnostic Prediction")

if submitted:
    input_array = np.array([input_vals])
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    st.write("---")
    st.subheader("Diagnostic Results")
    
    if prediction == 1:
        st.success(f"**Prediction: Benign (Low Risk)**")
        st.markdown(f"Confidence Score: **{probabilities[1]*100:.2f}%**")
    else:
        st.error(f"**Prediction: Malignant (High Risk / Concern Identified)**")
        st.markdown(f"Confidence Score: **{probabilities[0]*100:.2f}%**")
        
    st.info("Note: This tool is built for demonstration and educational purposes as part of a machine learning portfolio.")

# Sidebar info
with st.sidebar:
    st.header("About Project")
    st.markdown("**Task:** Disease Prediction from Medical Data")
    st.markdown("**Algorithm:** Random Forest Classifier")
    st.markdown("**Frameworks:** Python, Scikit-Learn, Streamlit")
    st.write("---")
    st.markdown("👨‍💻 Developed for Resume Portfolio Showcase.")