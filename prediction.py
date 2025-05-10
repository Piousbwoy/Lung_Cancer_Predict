import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import logging
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from utils.utils import generate_pdf_report  # For Download Report
import emoji  # Added for reliable emoji rendering

# ---------------------------------------
# Configure Logging
# ---------------------------------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------
# Configure File Paths
# ---------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "voting_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "lung_cancer_new.csv")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

# ---------------------------------------
# Color Scheme: 2025 Trends
# ---------------------------------------
COLORS = {
    "primary": "#3b82f6",       # Vibrant blue
    "primary_dark": "#1d4ed8",  # Darker blue
    "secondary": "#f0f9ff",     # Light blue
    "accent": "#14b8a6",        # Teal
    "accent_dark": "#0d9488",   # Dark teal
    "danger": "#ef4444",        # Red
    "success": "#10b981",       # Green
    "warning": "#f59e0b",       # Amber
    "gray": "#64748b",          # Slate
    "text": "#0f172a",          # Dark blue-black
    "text_light": "#475569",    # Light slate
    "background": "#ffffff",    # White
    "card": "#f8fafc",         # Off-white
    "glow": "#a5b4fc"          # Purple glow
}

# ---------------------------------------
# Gradient Definitions
# ---------------------------------------
GRADIENTS = {
    "primary": "linear-gradient(135deg, #3b82f6, #1d4ed8)",
    "accent": "linear-gradient(135deg, #14b8a6, #0d9488)",
    "success": "linear-gradient(135deg, #10b981, #059669)",
    "danger": "linear-gradient(135deg, #ef4444, #dc2626)",
    "warning": "linear-gradient(135deg, #f59e0b, #d97706)",
    "card": "linear-gradient(145deg, #f8fafc, #e2e8f0)",
    "header": "linear-gradient(135deg, #3b82f6, #14b8a6)",
    "button": "linear-gradient(90deg, #3b82f6, #9333ea)"  # Updated for vibrant buttons
}

# ---------------------------------------
# Mapping Dictionaries
# ---------------------------------------
gender_dict = {"Male": 1, "Female": 0}
feature_dict = {"No": 0, "Yes": 1}
prediction_label = {0: "Low Risk", 1: "High Risk"}
EXPECTED_FEATURES = 14

# ---------------------------------------
# Custom CSS: Enhanced Futuristic Design
# ---------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --primary: #3b82f6;
        --primary-dark: #1d4ed8;
        --secondary: #f0f9ff;
        --accent: #14b8a6;
        --accent-dark: #0d9488;
        --danger: #ef4444;
        --success: #10b981;
        --warning: #f59e0b;
        --gray: #64748b;
        --text: #0f172a;
        --text-light: #475569;
        --background: #ffffff;
        --card: #f8fafc;
        --glow: #a5b4fc;
    }

    .stApp {
        background-color: var(--background);
        font-family: 'Inter', sans-serif;
        color: var(--text);
        overflow-x: hidden;
        padding: 0;
        margin: 0;
    }

    .main-container {
        max-width: 1200px;
        margin: auto;
        padding: 30px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        line-height: 1.2;
        margin: 0;
        color: #000000; /* Black headings */
    }

    h2 { font-size: 3rem; margin: 2rem 0; }
    h3 { font-size: 2rem; margin: 1.5rem 0; }
    h4 { font-size: 1.5rem; margin: 1rem 0; }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #000000;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
    }

    .section-title::after {
        content: "";
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 120px;
        height: 5px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        border-radius: 3px;
    }

    .card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 15px;
        margin-bottom: 5px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
    }

    .chart-card {
        background: var(--card);
        border-radius: 20px;
        box-shadow: 8px 8px 16px rgba(0, 0, 0, 0.05),
                    -8px -8px 16px rgba(255, 255, 255, 0.8);
        padding: 10px;
        margin: 2px 0;
        transition: all 0.3s ease;
    }

    .chart-card:hover {
        box-shadow: 12px 12px 24px rgba(0, 0, 0, 0.07),
                   -12px -12px 24px rgba(255, 255, 255, 0.9);
    }

    .attribute-item {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: var(--text);
        border-bottom: 1px solid rgba(241, 245, 249, 0.5);
    }

    .attribute-label {
        font-weight: 600;
        color: var(--text);
    }

    .attribute-value {
        color: var(--primary);
        font-weight: 500;
    }

    .risk-item {
        background: rgba(248, 250, 252, 0.9);
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
        border-left: 5px solid var(--primary);
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }

    .risk-item:hover {
        background: var(--secondary);
        border-left-color: var(--accent);
        transform: translateX(5px);
    }

    .risk-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .risk-title.high-risk { color: var(--danger); }
    .risk-title.low-risk { color: var(--success); }

    .risk-item p {
        font-size: 1rem;
        color: var(--text-light);
        margin: 0;
    }

    .recommendation-item {
        display: flex;
        align-items: flex-start;
        padding: 15px 0;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: var(--text);
        border-bottom: 1px solid rgba(241, 245, 249, 0.5);
    }

    .recommendation-icon {
        font-size: 1.5rem;
        color: var(--accent);
        margin-right: 15px;
        margin-top: 3px;
    }

    .recommendation-item span {
        flex: 1;
        line-height: 1.6;
    }

    /* Enhanced button styling */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #9333ea);
        color: white;
        padding: 14px 28px;
        border: none;
        border-radius: 14px;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
        background: linear-gradient(90deg, #1d4ed8, #7e22ce);
    }

    .stButton>button:active {
        transform: scale(0.98);
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
    }

    .prediction-high-risk h2 { color: var(--danger); }
    .prediction-low-risk h2 { color: var(--success); }

    .disclaimer {
        font-size: 0.9rem;
        color: var(--gray);
        text-align: center;
        margin-top: 25px;
        line-height: 1.6;
    }

    .explanation-text {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: var(--text-light);
        line-height: 1.8;
        margin: 15px 0;
    }

    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 240px;
        background-color: var(--primary-dark);
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 10px;
        position: absolute;
        z-index: 10;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    .fade-in { animation: fadeIn 1s ease-in; }

    @keyframes breathing {
        0% { box-shadow: 0 0 10px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.6); }
        100% { box-shadow: 0 0 10px rgba(59, 130, 246, 0.3); }
    }

    .breathing { animation: breathing 4s infinite ease-in-out; }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .pulse { animation: pulse 2s infinite ease-in-out; }

    .progress-container {
        width: 100%;
        height: 8px;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 4px;
        margin: 15px 0;
    }

    .progress-bar {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #3b82f6, #9333ea);
        transition: width 0.1s ease-in-out;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: var(--text);
        padding: 12px 24px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--secondary);
        color: var(--primary);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    .stNumberInput, .stRadio {
        background: var(--card);
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 15px;
        font-family: 'Inter', sans-serif;
    }

    .stNumberInput input, .stRadio label {
        color: var(--text);
        font-size: 1rem;
    }

    @media (max-width: 1024px) {
        .main-container { padding: 20px; }
        h2 { font-size: 2.5rem; }
        h3 { font-size: 1.75rem; }
        .section-title { font-size: 2rem; }
        .card { padding: 20px; }
    }

    @media (max-width: 768px) {
        h2 { font-size: 2rem; }
        h3 { font-size: 1.5rem; }
        .section-title { font-size: 1.75rem; }
        .card { padding: 15px; }
        .attribute-item { font-size: 1rem; }
        .risk-item { padding: 12px; }
        .stButton>button { padding: 12px 20px; font-size: 1rem; }
    }

    @media (max-width: 480px) {
        h2 { font-size: 1.75rem; }
        h3 { font-size: 1.25rem; }
        .section-title { font-size: 1.5rem; }
        .main-container { padding: 15px; }
        .card { padding: 10px; }
        .attribute-item { font-size: 0.9rem; }
        .risk-item { padding: 10px; }
        .stButton>button { padding: 10px 16px; font-size: 0.9rem; }
        .tooltip .tooltiptext { width: 180px; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Background SVG
# ---------------------------------------
BACKGROUND_SVG = """
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; opacity: 0.05;">
    <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <pattern id="pattern" x="0" y="0" width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M0 25 L50 25 M25 0 L25 50" stroke="#3b82f6" stroke-width="0.8"/>
            <circle cx="25" cy="25" r="1.5" fill="#3b82f6" />
        </pattern>
        <rect width="100%" height="100%" fill="url(#pattern)" />
    </svg>
</div>
"""

# ---------------------------------------
# Helper Functions
# ---------------------------------------
def get_value(val, my_dict):
    """Maps a string value to its numeric code."""
    return my_dict.get(val)

def get_fvalue(val):
    """Maps a feature value (Yes/No) to binary."""
    return feature_dict.get(val)

def load_model_and_scaler(model_file, scaler_file):
    """
    Loads the trained model and scaler.
    """
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file) if os.path.exists(scaler_file) else None
        logging.info(f"Loaded model from {model_file} and scaler from {scaler_file}")
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        logging.error(f"File not found: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading model/scaler: {e}")
        st.error(f"Error loading model/scaler: {e}")
        return None, None

def preprocess_features(feature_list, scaler):
    """
    Preprocesses input features for prediction.
    """
    if len(feature_list) != EXPECTED_FEATURES:
        raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {len(feature_list)}")
    features = np.array(feature_list).reshape(1, -1)
    numerical_indices = [0, 8, 9, 10, 11, 12]
    if scaler:
        features[:, numerical_indices] = scaler.transform(features[:, numerical_indices])
    return features

def create_dual_gauge_chart(high_risk, low_risk):
    """
    Creates a dual gauge chart for risk probabilities.
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]]
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=high_risk,
            title={'text': "High Risk Probability", 'font': {'size': 16, 'color': COLORS["text"]}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS["text"]},
                'bar': {'color': COLORS["danger"]},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': COLORS["gray"],
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ],
            }
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=low_risk,
            title={'text': "Low Risk Probability", 'font': {'size': 16, 'color': COLORS["text"]}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS["text"]},
                'bar': {'color': COLORS["success"]},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': COLORS["gray"],
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.2)'},
                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                ],
            }
        ),
        row=1, col=2
    )
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text"], 'family': "Inter"},
        showlegend=False
    )
    return fig

def create_patient_radar_chart(feature_list):
    """
    Creates a radar chart comparing patient features to averages.
    """
    categories = [
        "Age", "Smoking", "Cough", "Fatigue", "Blood",
        "Pain", "Weight Loss", "Tumor Size", "Lung Function", "Tumor Marker"
    ]
    patient_values = [
        min(100, feature_list[0] * 1.5),
        100 if feature_list[2] == 1 else 0,
        100 if feature_list[3] == 1 else 0,
        100 if feature_list[4] == 1 else 0,
        100 if feature_list[5] == 1 else 0,
        100 if feature_list[6] == 1 else 0,
        100 if feature_list[7] == 1 else 0,
        min(100, feature_list[8] * 20),
        min(100, feature_list[11] * 20),
        min(100, feature_list[12] * 2)
    ]
    population_values = [50, 30, 25, 20, 10, 15, 15, 20, 50, 10]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=patient_values,
        theta=categories,
        fill='toself',
        name='You',
        line_color=COLORS["primary"],
        fillcolor=f'rgba(59, 130, 246, 0.3)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=population_values,
        theta=categories,
        fill='toself',
        name='Average',
        line_color=COLORS["gray"],
        fillcolor=f'rgba(100, 116, 139, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=12))
        ),
        showlegend=True,
        title=dict(
            text="Your Risk Profile",
            font=dict(size=20, family="Space Grotesk", color=COLORS["text"]),
            x=0.5
        ),
        height=450,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text"], 'family': "Inter"},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

def show_patient_summary(feature_list):
    """Displays a summary of patient input features."""
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Your Input Summary</h3>', unsafe_allow_html=True)
    labels = [
        "Age", "Sex", "Smoking History", "Persistent Cough",
        "Fatigue", "Coughing Up Blood", "Chest Pain", "Weight Loss",
        "Tumor Size (cm)", "Alk Phosphate", "SGOT",
        "Lung Function (%)", "Tumor Marker", "Histology"
    ]
    display_values = [
        str(feature_list[0]),
        "Male" if feature_list[1] == 1 else "Female",
        "Yes" if feature_list[2] == 1 else "No",
        "Yes" if feature_list[3] == 1 else "No",
        "Yes" if feature_list[4] == 1 else "No",
        "Yes" if feature_list[5] == 1 else "No",
        "Yes" if feature_list[6] == 1 else "No",
        "Yes" if feature_list[7] == 1 else "No",
        str(feature_list[8]),
        str(feature_list[9]),
        str(feature_list[10]),
        str(feature_list[11]),
        str(feature_list[12]),
        "Abnormal" if feature_list[13] == 1 else "Normal"
    ]
    cols = st.columns(2)
    for i, (label, value) in enumerate(zip(labels, display_values)):
        with cols[i % 2]:
            st.markdown(f"""
                <div class="attribute-item">
                    <div class="attribute-label">{label}</div>
                    <div class="attribute-value">{value}</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_risk_factors(feature_list):
    """Displays key risk factors based on inputs."""
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Key Risk Factors</h3>', unsafe_allow_html=True)
    risk_factors = []
    if feature_list[2] == 1:
        risk_factors.append(("Smoking History", "Smoking is the leading cause of lung cancer.", "high-risk"))
    else:
        risk_factors.append(("Smoking History", "No smoking history reduces your risk.", "low-risk"))
    if feature_list[0] > 55:
        risk_factors.append(("Age", "Risk increases significantly after age 55.", "high-risk"))
    else:
        risk_factors.append(("Age", f"Age {feature_list[0]} is within lower-risk range.", "low-risk"))
    if feature_list[5] == 1:
        risk_factors.append(("Coughing Blood", "Hemoptysis may indicate tumor presence.", "high-risk"))
    else:
        risk_factors.append(("Coughing Blood", "No hemoptysis reported.", "low-risk"))
    if feature_list[6] == 1:
        risk_factors.append(("Chest Pain", "Persistent pain can signal tumor growth.", "high-risk"))
    else:
        risk_factors.append(("Chest Pain", "No chest pain reported.", "low-risk"))
    if feature_list[7] == 1:
        risk_factors.append(("Weight Loss", "Unexplained weight loss is a concerning symptom.", "high-risk"))
    else:
        risk_factors.append(("Weight Loss", "No weight loss reported.", "low-risk"))
    if feature_list[8] > 3:
        risk_factors.append(("Tumor Size", f"A {feature_list[8]:.1f} cm tumor suggests advanced disease.", "high-risk"))
    else:
        risk_factors.append(("Tumor Size", f"Tumor size {feature_list[8]:.1f} cm is less concerning.", "low-risk"))
    if feature_list[12] > 10:
        risk_factors.append(("Tumor Marker", f"Elevated marker ({feature_list[12]:.1f} μg/L) indicates risk.", "high-risk"))
    else:
        risk_factors.append(("Tumor Marker", f"Tumor marker {feature_list[12]:.1f} μg/L is normal.", "low-risk"))
    if feature_list[11] < 2:
        risk_factors.append(("Lung Function", "Reduced function suggests severe disease.", "high-risk"))
    else:
        risk_factors.append(("Lung Function", f"Lung function {feature_list[11]:.1f}% is adequate.", "low-risk"))
    if feature_list[13] == 1:
        risk_factors.append(("Histology", "Abnormal biopsy confirms malignancy.", "high-risk"))
    else:
        risk_factors.append(("Histology", "Normal histology reduces concern.", "low-risk"))
    if not any(rf[2] == "high-risk" for rf in risk_factors):
        risk_factors.append(("Overall Risk", "No major risk factors detected.", "low-risk"))
    for title, description, risk_class in risk_factors:
        st.markdown(f"""
            <div class="risk-item">
                <div class="risk-title {risk_class}">{title}</div>
                <p>{description}</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <p class="explanation-text">
            These factors are based on your inputs and medical patterns. Consult a pulmonologist for a full evaluation.
        </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_treatment_recommendations(prediction):
    """Displays recommended next steps."""
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Recommended Next Steps</h3>', unsafe_allow_html=True)
    recommendations = [
        (emoji.emojize(":clipboard:"),"Urgent CT scan to evaluate lung abnormalities.") if prediction == "High Risk" else (emoji.emojize(":clipboard:"), "Consider annual low-dose CT screening if a smoker."),
        (emoji.emojize(":stethoscope:"),"Suggested Drugs: like <strong>Osimertinib (for EGFR mutations), Alectinib (for ALK rearrangements), and Lorlatinib (for ROS1 fusions).</strong>") if prediction == "High Risk" else (emoji.emojize(":magnifying_glass_tilted_left:"), "Monitor symptoms like cough or fatigue."),
        (emoji.emojize(":test_tube:"),"Blood tests for tumor markers (e.g., CEA, CYFRA 21-1).") if prediction == "High Risk" else (emoji.emojize(":deciduous_tree:"), "Quit smoking to reduce risk."),
        (emoji.emojize(":hospital:"),"Discuss treatment options (surgery, chemo, radiation).") if prediction == "High Risk" else (emoji.emojize(":lungs:"), "Regular spirometry to assess lung function."),
        (emoji.emojize(":calendar:"),"Follow up within 1–2 weeks.") if prediction == "High Risk" else (emoji.emojize(":mobile_phone:"), "Learn warning signs requiring immediate attention.")
    ]
    for idx, (icon, text) in enumerate(recommendations, 1):
        st.markdown(f"""
            <div class="recommendation-item">
                <span class="recommendation-icon">{icon} {idx}.</span>
                <span>{text}</span>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <p class="explanation-text">
            These recommendations are preliminary. A healthcare provider will tailor advice to your case.
        </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_prediction_page():
    """
    Renders the lung cancer risk assessment page.
    """
    st.markdown(BACKGROUND_SVG, unsafe_allow_html=True)
    st.markdown('<div class="main-container breathing">', unsafe_allow_html=True)
    st.markdown("""
        <h2 class="section-title pulse">
            Lung Cancer Mortality Risk Assessment
        </h2>
    """, unsafe_allow_html=True)

    # Initialize session state
    session_keys = [
        "patient_data", "prediction_result", "prediction_probs",
        "submission_id", "username", "active_tab", "show_view_results"
    ]
    for key in session_keys:
        if key not in st.session_state:
            if key == "submission_id":
                st.session_state[key] = 0
            elif key == "show_view_results":
                st.session_state[key] = False
            else:
                st.session_state[key] = None
    if st.session_state.active_tab is None:
        st.session_state.active_tab = "Enter Data"

    tab1, tab2 = st.tabs(["Enter Data", "Results"])

    with tab1:
        st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Enter Your Health Details</h3>', unsafe_allow_html=True)
        st.markdown("""
            <p class="explanation-text">
                Provide accurate health information to assess your lung cancer mortality risk. 
                All fields are used to generate a personalized prediction.
            </p>
        """, unsafe_allow_html=True)

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                age = st.number_input(
                    "Age",
                    20, 95, 55,
                    help="Your age in years",
                    key="age"
                )
                st.markdown('<span class="tooltiptext">Enter your current age (20-95 years).</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                sex = st.radio(
                    "Sex",
                    list(gender_dict.keys()),
                    key="sex"
                )
                st.markdown('<span class="tooltiptext">Select your biological sex.</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                smoking = st.radio(
                    "Smoking History",
                    list(feature_dict.keys()),
                    index=1,
                    help="Have you smoked regularly?",
                    key="smoking"
                )
                st.markdown('<span class="tooltiptext">Indicate if you have a history of regular smoking.</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                persistent_cough = st.radio(
                    "Persistent Cough",
                    list(feature_dict.keys()),
                    index=0,
                    key="cough"
                )
                st.markdown('<span class="tooltiptext">Do you have a cough lasting over 3 weeks?</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                fatigue = st.radio(
                    "Fatigue",
                    list(feature_dict.keys()),
                    index=0,
                    key="fatigue"
                )
                st.markdown('<span class="tooltiptext">Do you experience unusual tiredness?</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                cough_blood = st.radio(
                    "Coughing Up Blood",
                    list(feature_dict.keys()),
                    index=0,
                    key="blood"
                )
                st.markdown('<span class="tooltiptext">Have you coughed up blood recently?</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                chest_pain = st.radio(
                    "Chest Pain",
                    list(feature_dict.keys()),
                    index=0,
                    key="pain"
                )
                st.markdown('<span class="tooltiptext">Do you have persistent chest pain?</span></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                weight_loss = st.radio(
                    "Unexplained Weight Loss",
                    list(feature_dict.keys()),
                    index=0,
                    key="weight_loss"
                )
                st.markdown('<span class="tooltiptext">Have you lost weight without trying?</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                histology = st.radio(
                    "Abnormal Histology",
                    list(feature_dict.keys()),
                    index=0,
                    help="Biopsy result, if known",
                    key="histology"
                )
                st.markdown('<span class="tooltiptext">Select if biopsy shows abnormal cells (if known).</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                tumor_size = st.number_input(
                    "Tumor Size (cm)",
                    0.0, 8.0, 1.0,
                    step=0.1,
                    help="From imaging, if available",
                    key="tumor_size"
                )
                st.markdown('<span class="tooltiptext">Enter tumor size from recent imaging (0-8 cm).</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                alk_phosphate = st.number_input(
                    "Alkaline Phosphatase (IU/L)",
                    40.0, 296.0, 85.0,
                    step=1.0,
                    key="alk_phosphate"
                )
                st.markdown('<span class="tooltiptext">Blood test result (normal range: 40-296 IU/L).</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                sgot = st.number_input(
                    "SGOT (U/L)",
                    10.0, 648.0, 45.0,
                    step=1.0,
                    key="sgot"
                )
                st.markdown('<span class="tooltiptext">Liver enzyme level (normal range: 10-648 U/L).</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                lung_function = st.number_input(
                    "Lung Function (% FEV1)",
                    0.5, 5.0, 2.8,
                    step=0.1,
                    help="From spirometry",
                    key="lung_function"
                )
                st.markdown('<span class="tooltiptext">Forced expiratory volume from spirometry (0.5-5.0%).</span></div>', unsafe_allow_html=True)
                
                st.markdown('<div class="tooltip">', unsafe_allow_html=True)
                tumor_marker = st.number_input(
                    "Tumor Marker (μg/L)",
                    0.0, 100.0, 5.0,
                    step=1.0,
                    help="E.g., CEA level",
                    key="tumor_marker"
                )
                st.markdown('<span class="tooltiptext">Blood marker like CEA (normal <10 μg/L).</span></div>', unsafe_allow_html=True)

            submitted = st.form_submit_button("Run Assessment", type="primary")
        
        if st.button("Reset Assessment", key="reset_button"):
            st.session_state.patient_data = None
            st.session_state.prediction_result = None
            st.session_state.prediction_probs = None
            st.session_state.submission_id = 0
            st.session_state.username = None
            st.session_state.show_view_results = False
            st.session_state.active_tab = "Enter Data"
            st.success("Assessment reset. Enter new data.")
            logging.info("Reset submission_id to 0")
        
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            if st.session_state.submission_id is None:
                st.session_state.submission_id = 0
                logging.warning("submission_id was None during submission; set to 0")
            st.session_state.submission_id += 1
            current_submission_id = st.session_state.submission_id
            st.session_state.prediction_result = None
            st.session_state.prediction_probs = None
            st.session_state.show_view_results = False
            logging.info(f"Incremented submission_id to {current_submission_id}")

            try:
                sex_val = get_value(sex, gender_dict)
                smoking_val = get_fvalue(smoking)
                persistent_cough_val = get_fvalue(persistent_cough)
                fatigue_val = get_fvalue(fatigue)
                cough_blood_val = get_fvalue(cough_blood)
                chest_pain_val = get_fvalue(chest_pain)
                weight_loss_val = get_fvalue(weight_loss)
                histology_val = get_fvalue(histology)
                feature_list = [
                    age, sex_val, smoking_val, persistent_cough_val,
                    fatigue_val, cough_blood_val, chest_pain_val,
                    weight_loss_val, tumor_size, alk_phosphate, sgot,
                    lung_function, tumor_marker, histology_val
                ]
                logging.info(f"Submission {current_submission_id} - Input features: {feature_list}")

                if len(feature_list) != EXPECTED_FEATURES:
                    st.error(f"Invalid input: Expected {EXPECTED_FEATURES} features, got {len(feature_list)}")
                    logging.error(f"Submission {current_submission_id} - Feature count mismatch")
                    return

                st.session_state.patient_data = feature_list

                with st.spinner("Analyzing your data..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.002)  # Faster for better UX
                        progress_bar.progress(i + 1)
                    progress_bar.empty()

                    logging.info("Attempting to load model and scaler")
                    if not callable(load_model_and_scaler):
                        st.error("Internal error: Model loading function not found.")
                        logging.error("load_model_and_scaler is not callable.")
                        return

                    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
                    if model:
                        single_sample = preprocess_features(feature_list, scaler)
                        logging.info(f"Submission {current_submission_id} - Preprocessed features: {single_sample}")
                        prediction = model.predict(single_sample)
                        prediction_proba = model.predict_proba(single_sample)
                        logging.info(f"Submission {current_submission_id} - Prediction: {prediction}, Probabilities: {prediction_proba}")
                        if st.session_state.submission_id == current_submission_id:
                            st.session_state.prediction_result = prediction_label[int(prediction[0])]
                            st.session_state.prediction_probs = {
                                "High Risk": round(prediction_proba[0][1] * 100, 1),
                                "Low Risk": round(prediction_proba[0][0] * 100, 1)
                            }
                            st.session_state.show_view_results = True
                            st.session_state.active_tab = "Results"  # Switch to Results tab
                            st.success("Analysis complete! Results are ready.")
                        else:
                            logging.warning(f"Submission {current_submission_id} discarded due to newer submission")
                            st.warning("A newer assessment was started. Check the latest results.")
                    else:
                        st.error("Unable to load prediction model.")
                        return
            except Exception as e:
                logging.error(f"Submission {current_submission_id} - Prediction error: {e}")
                st.error(f"Error processing prediction: {e}")
                return

    with tab2:
        if (st.session_state.prediction_result 
            and st.session_state.patient_data 
            and st.session_state.prediction_probs 
            and st.session_state.submission_id is not None
            and len(st.session_state.patient_data) == EXPECTED_FEATURES):
            result = st.session_state.prediction_result
            probs = st.session_state.prediction_probs
            feature_list = st.session_state.patient_data

            st.markdown(f"""
                <div class="card fade-in prediction-{'high-risk' if result == 'High Risk' else 'low-risk'} breathing">
                    <h3 class="section-title">Your Risk Assessment</h3>
                    <h2 style="color: {'#ef4444' if result == 'High Risk' else '#10b981'}; text-align: center;">
                        {result}
                    </h2>
                    <p class="explanation-text" style="text-align: center;">
                    {"The model indicates a <b>high risk of mortality</b> from lung cancer. Urgent consultation with an oncologist is essential." if result == 'High Risk' else "The model predicts a <b>low mortality risk</b> from lung cancer. Maintain routine check-ups with your doctor."}
                </p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="chart-card fade-in">', unsafe_allow_html=True)
            gauge_fig = create_dual_gauge_chart(probs["High Risk"], probs["Low Risk"])
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.markdown("""
                <p class="explanation-text">
                    These gauges show the likelihood of high and low mortality risk, 
                    with percentages reflecting model confidence. A high-risk score above 70% suggests urgent follow-up; 
                    low-risk scores indicate a better prognosis but require continued monitoring.
                </p>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            result_tab1, result_tab2, result_tab3 = st.tabs(["Summary", "Risk Factors", "Next Steps"])
            with result_tab1:
                show_patient_summary(feature_list)
                st.markdown('<div class="chart-card fade-in">', unsafe_allow_html=True)
                st.markdown('<h4 class="section-title">Your Risk Profile</h4>', unsafe_allow_html=True)
                radar_fig = create_patient_radar_chart(feature_list)
                st.plotly_chart(radar_fig, use_container_width=True)
                st.markdown("""
                    <p class="explanation-text">
                        This radar chart compares your health metrics to population averages. 
                        Larger spikes (e.g., smoking, tumor size) highlight elevated risk factors, 
                        guiding patients to focus on key areas and clinicians to prioritize diagnostics.
                    </p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_tab2:
                show_risk_factors(feature_list)
            
            with result_tab3:
                show_treatment_recommendations(result)
                st.markdown("""
                    <p class="disclaimer">
                        <strong>Disclaimer:</strong> This tool provides risk estimates, not a diagnosis. 
                        Consult a healthcare professional for medical advice.
                    </p>
                """, unsafe_allow_html=True)
            
            # Download Report Section
            st.markdown('<div class="card fade-in breathing">', unsafe_allow_html=True)
            st.markdown('<h4 class="section-title">Download Your Report</h4>', unsafe_allow_html=True)
            if st.button("Download Report", key="download_report", type="primary"):
                try:
                    pdf_buffer = generate_pdf_report(
                        username=st.session_state.username or "User",
                        prediction_result=result,
                        prediction_probs=probs,
                        feature_list=feature_list
                    )
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"PulmoPredict_Report_{st.session_state.username or 'User'}_{st.session_state.submission_id}.pdf",
                        mime="application/pdf",
                        key="download_pdf",
                        type="primary"
                    )
                except Exception as e:
                    logging.error(f"PDF generation error: {e}")
                    st.error("Error generating report. Please try again.")
            st.markdown("""
                <p class="disclaimer">
                    <strong>Disclaimer:</strong> This report is for informational purposes only. 
                    Share it with your healthcare provider for professional evaluation.
                </p>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.markdown("""
                <div class="card fade-in">
                    <p class="explanation-text" style="text-align: center;">
                        No results available. Please submit your health details in the 'Enter Data' tab to view your mortality risk assessment.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show_prediction_page()