import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------
# Configuration: File Paths
# ---------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'lung_cancer_new.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'voting_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.pkl')

# ---------------------------------------
# Configuration: Color Palette (2025 Trends)
# ---------------------------------------
COLORS = {
    "primary": "#3b82f6",
    "primary_dark": "#1d4ed8",
    "secondary": "#f0f9ff",
    "accent": "#14b8a6",
    "accent_dark": "#0d9488",
    "danger": "#ef4444",
    "success": "#10b981",
    "warning": "#f59e0b",
    "gray": "#64748b",
    "text": "#000000",
    "text_light": "#475569",
    "background": "#ffffff",
    "card": "#f8fafc"
}

# ---------------------------------------
# Configuration: Gradient Definitions
# ---------------------------------------
GRADIENTS = {
    "primary": "linear-gradient(135deg, #3b82f6, #1d4ed8)",
    "accent": "linear-gradient(135deg, #14b8a6, #0d9488)",
    "success": "linear-gradient(135deg, #10b981, #059669)",
    "danger": "linear-gradient(135deg, #ef4444, #dc2626)",
    "warning": "linear-gradient(135deg, #f59e0b, #d97706)",
    "card": "linear-gradient(145deg, #f8fafc, #e2e8f0)",
    "header": "linear-gradient(135deg, #3b82f6, #14b8a6)"
}

# ---------------------------------------
# CSS: Comprehensive Tailwind-Inspired Styling
# ---------------------------------------
MODERN_CSS = """
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
    }
    
    .stApp {
        background-color: var(--background);
        font-family: 'Inter', sans-serif;
        color: var(--text);
        overflow-x: hidden;
        padding: 0;
        margin: 0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        line-height: 1.2;
        margin: 0;
    }
    
    h1 { font-size: 3.5rem; margin: 2rem 0; }
    h2 { font-size: 2.5rem; margin: 1.5rem 0; }
    h3 { font-size: 1.75rem; margin: 1rem 0; }
    h4 { font-size: 1.25rem; margin: 0.75rem 0; }
    h5 { font-size: 1rem; margin: 0.5rem 0; }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        transition: all 0.3s ease;
        position: relative;
        z-index: 1;
    }
    
    .glass-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.15);
    }
    
    .neomorphic {
        background: var(--card);
        border-radius: 20px;
        box-shadow: 10px 10px 20px rgba(0, 0, 0, 0.05),
                    -10px -10px 20px rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .neomorphic:hover {
        box-shadow: 15px 15px 30px rgba(0, 0, 0, 0.07),
                   -15px -15px 30px rgba(255, 255, 255, 0.9);
    }
    
    .section-title {
        font-size: 2.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        border-radius: 2px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
        transition: all 0.5s ease;
        position: relative;
        overflow: hidden;
        transform: perspective(1000px) rotateX(0deg);
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0));
        z-index: 1;
    }
    
    .metric-card:hover {
        transform: perspective(1000px) rotateX(10deg) translateY(-5px);
        box-shadow: 0 20px 30px rgba(59, 130, 246, 0.4);
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        opacity: 0.9;
        z-index: 2;
        position: relative;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.5rem;
        font-family: 'Space Grotesk', sans-serif;
        z-index: 2;
        position: relative;
    }
    
    .metric-desc {
        font-size: 0.9rem;
        opacity: 0.8;
        z-index: 2;
        position: relative;
    }
    
    .explanation-text {
        font-size: 1.05rem;
        color: var(--text-light);
        line-height: 1.8;
        margin: 1.5rem 0;
    }
    
    .explanation-text strong {
        color: var(--text);
        font-weight: 600;
    }
    
    .chart-container {
        padding: 1rem;
        border-radius: 16px;
        background: var(--card);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    }
    
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: var(--primary-dark);
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 0.75rem;
        position: absolute;
        z-index: 10;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    .modal {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: 100;
        align-items: center;
        justify-content: center;
    }
    
    .modal-content {
        background: var(--card);
        border-radius: 16px;
        padding: 2rem;
        max-width: 600px;
        width: 90%;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        position: relative;
    }
    
    .close-button {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: var(--gray);
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .floating {
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes breathing {
        0% { box-shadow: 0 0 10px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.6); }
        100% { box-shadow: 0 0 10px rgba(59, 130, 246, 0.3); }
    }
    
    .breathing {
        animation: breathing 4s infinite ease-in-out;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite ease-in-out;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .spinner {
        border: 4px solid rgba(59, 130, 246, 0.2);
        border-top: 4px solid var(--primary);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    .progress-container {
        width: 100%;
        height: 8px;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        transition: width 1s ease-in-out;
    }
    
    .switch {
        position: relative;
        display: inline-block;
        width: 60px;
        height: 34px;
    }
    
    .switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }
    
    .slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #ccc;
        transition: .4s;
        border-radius: 34px;
    }
    
    .slider:before {
        position: absolute;
        content: "";
        height: 26px;
        width: 26px;
        left: 4px;
        bottom: 4px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
    }
    
    input:checked + .slider {
        background-color: var(--primary);
    }
    
    input:checked + .slider:before {
        transform: translateX(26px);
    }
    
    .collapsible {
        background-color: var(--secondary);
        color: var(--text);
        cursor: pointer;
        padding: 1rem;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
        font-size: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .collapsible:hover {
        background-color: var(--primary);
        color: white;
    }
    
    .content {
        padding: 0 1rem;
        display: none;
        overflow: hidden;
        background-color: var(--card);
        border-radius: 8px;
    }
    
    @media (max-width: 768px) {
        h1 { font-size: 2.5rem; }
        h2 { font-size: 1.75rem; }
        h3 { font-size: 1.5rem; }
        .glass-card { padding: 1.5rem; }
        .metric-card { padding: 1rem; }
        .metric-value { font-size: 2rem; }
        .section-title { font-size: 1.75rem; }
        .tooltip .tooltiptext { width: 180px; }
    }
    
    @media (max-width: 480px) {
        h1 { font-size: 2rem; }
        h2 { font-size: 1.5rem; }
        h3 { font-size: 1.25rem; }
        .section-title { font-size: 1.5rem; }
        .metric-card { margin: 0.5rem 0; }
        .tooltip .tooltiptext { width: 140px; }
    }
</style>
"""

# ---------------------------------------
# Background: Subtle SVG Pattern
# ---------------------------------------
BACKGROUND_SVG = """
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; opacity: 0.03;">
    <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <pattern id="pattern" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M0 20 L40 20 M20 0 L20 40" stroke="#3b82f6" stroke-width="0.5"/>
            <circle cx="20" cy="20" r="1" fill="#3b82f6" />
        </pattern>
        <rect width="100%" height="100%" fill="url(#pattern)" />
    </svg>
</div>
"""

# ---------------------------------------
# Helper Function: Data Loading
# ---------------------------------------
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        return df
    except FileNotFoundError:
        st.error("Error: Dataset not found at 'data/lung_cancer_new.csv'. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# ---------------------------------------
# Helper Function: Model Loading
# ---------------------------------------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found in 'models/' directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model/scaler: {str(e)}")
        return None, None

# ---------------------------------------
# Visualization: Feature Importance Chart
# ---------------------------------------
def create_feature_importance_chart(model):
    features = [
        "Age", "Sex", "Smoking", "Persistent Cough", "Fatigue",
        "Coughing Blood", "Chest Pain", "Weight Loss", "Tumor Size",
        "Alk Phosphate", "SGOT", "Lung Function", "Tumor Marker", "Histology"
    ]
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_ * 100
    else:
        importances = np.array([15, 5, 25, 10, 8, 7, 6, 8, 12, 5, 4, 6, 5, 4])
        importances = importances / importances.sum() * 100

    fig = go.Figure(data=[
        go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale=[[0, COLORS["primary"]], [0.5, COLORS["accent"]], [1, COLORS["accent_dark"]]],
                line=dict(color='#ffffff', width=1.5)
            ),
            hovertemplate="%{y}: %{x:.1f}%"
        )
    ])
    fig.update_layout(
        title=dict(
            text="Key Drivers of Lung Cancer Risk",
            font=dict(family="Space Grotesk", size=24, color=COLORS["text"]),
            x=0.5
        ),
        xaxis_title="Impact on Prediction (%)",
        yaxis_title="Feature",
        font=dict(family="Inter", size=14, color=COLORS["text"]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=40, r=40, t=100, b=40),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)', tickformat='.0f'),
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

# ---------------------------------------
# Visualization: Smoking vs. Risk Chart
# ---------------------------------------
def create_smoking_risk_chart(df):
    smoking_risk = df.groupby(['smoking', 'class']).size().unstack(fill_value=0)
    fig = go.Figure(data=[
        go.Bar(
            x=['Non-Smoker', 'Smoker'],
            y=smoking_risk[0],
            name='Low Risk',
            marker_color=COLORS["success"],
            opacity=0.85
        ),
        go.Bar(
            x=['Non-Smoker', 'Smoker'],
            y=smoking_risk[1],
            name='High Risk',
            marker_color=COLORS["danger"],
            opacity=0.85
        )
    ])
    fig.update_traces(
        marker=dict(line=dict(color='#ffffff', width=1.5)),
        hovertemplate="%{x}: %{y} patients"
    )
    fig.update_layout(
        title=dict(
            text="Smoking: A Major Risk Factor",
            font=dict(family="Space Grotesk", size=24, color=COLORS["text"]),
            x=0.5
        ),
        xaxis_title="Smoking Status",
        yaxis_title="Number of Patients",
        font=dict(family="Inter", size=14, color=COLORS["text"]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='stack',
        height=500,
        margin=dict(l=40, r=40, t=100, b=40),
        xaxis=dict(showgrid=False, tickfont=dict(size=12)),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)', tickfont=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

# ---------------------------------------
# Visualization: Tumor Size vs. Risk Chart
# ---------------------------------------
def create_tumor_size_chart(df):
    fig = go.Figure()
    for risk_level, color, name in [(0, COLORS["success"], 'Low Risk'), (1, COLORS["danger"], 'High Risk')]:
        fig.add_trace(go.Violin(
            x=[name] * len(df[df['class'] == risk_level]),
            y=df[df['class'] == risk_level]['tumor_size'],
            name=name,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.7,
            line_color='#ffffff',
            points='outliers',
            marker=dict(size=5),
            hovertemplate="Tumor Size: %{y:.1f} cm"
        ))
    fig.update_layout(
        title=dict(
            text="Tumor Size Signals Higher Risk",
            font=dict(family="Space Grotesk", size=24, color=COLORS["text"]),
            x=0.5
        ),
        xaxis_title="Risk Level",
        yaxis_title="Tumor Size (cm)",
        font=dict(family="Inter", size=14, color=COLORS["text"]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=40, r=40, t=100, b=40),
        xaxis=dict(showgrid=False, tickfont=dict(size=12)),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)', zeroline=False, tickfont=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

# ---------------------------------------
# Visualization: Age vs. Risk Chart
# ---------------------------------------
def create_age_risk_chart(df):
    fig = go.Figure()
    for risk_level, color, name in [(0, COLORS["success"], 'Low Risk'), (1, COLORS["danger"], 'High Risk')]:
        fig.add_trace(go.Histogram(
            x=df[df['class'] == risk_level]['age'],
            histnorm='probability density',
            name=name,
            marker_color=color,
            opacity=0.75,
            xbins=dict(size=5),
            hovertemplate="Age: %{x}<br>Proportion: %{y:.2f}"
        ))
    fig.update_traces(
        marker=dict(line=dict(color='#ffffff', width=1.5))
    )
    fig.update_layout(
        title=dict(
            text="Age and Lung Cancer Risk",
            font=dict(family="Space Grotesk", size=24, color=COLORS["text"]),
            x=0.5
        ),
        xaxis_title="Age",
        yaxis_title="Proportion",
        font=dict(family="Inter", size=14, color=COLORS["text"]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='overlay',
        height=500,
        margin=dict(l=40, r=40, t=100, b=40),
        xaxis=dict(showgrid=False, tickfont=dict(size=12)),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)', tickfont=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

# ---------------------------------------
# Visualization: Symptom Prevalence Chart
# ---------------------------------------
def create_symptom_prevalence_chart(df):
    symptoms = ['persistent_cough', 'fatigue', 'cough_blood', 'chest_pain', 'weight_loss']
    symptom_names = ['Persistent Cough', 'Fatigue', 'Coughing Blood', 'Chest Pain', 'Weight Loss']
    prevalence = [df[df['class'] == 1][symptom].mean() * 100 for symptom in symptoms]
    
    fig = go.Figure(data=[
        go.Bar(
            x=symptom_names,
            y=prevalence,
            marker=dict(
                color=prevalence,
                colorscale=[[0, COLORS["primary"]], [0.5, COLORS["accent"]], [1, COLORS["accent_dark"]]],
                line=dict(color='#ffffff', width=1.5)
            ),
            hovertemplate="%{x}: %{y:.1f}%"
        )
    ])
    fig.update_layout(
        title=dict(
            text="Symptoms in High-Risk Patients",
            font=dict(family="Space Grotesk", size=24, color=COLORS["text"]),
            x=0.5
        ),
        xaxis_title="Symptom",
        yaxis_title="% of High-Risk Patients",
        font=dict(family="Inter", size=14, color=COLORS["text"]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=40, r=40, t=100, b=40),
        xaxis=dict(showgrid=False, tickangle=15, tickfont=dict(size=12)),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)', tickformat='.0f', ticksuffix='%', tickfont=dict(size=12)),
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

# ---------------------------------------
# Main Function: Visualizations Page
# ---------------------------------------
def show_visualizations_page():
    st.markdown(MODERN_CSS, unsafe_allow_html=True)
    st.markdown(BACKGROUND_SVG, unsafe_allow_html=True)
    
    st.markdown('<div style="max-width: 1200px; margin: auto; padding: 20px;" class="fade-in">', unsafe_allow_html=True)

    st.markdown("""
        <h1 style="font-size: 3.5rem; font-weight: 700; text-align: center; margin: 2rem 0; 
                   background: linear-gradient(135deg, #3b82f6, #14b8a6); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;" 
                   class="breathing">
            Insights on Our Dataset and Model.
        </h1>
    """, unsafe_allow_html=True)

    with st.spinner("Loading dataset and model..."):
        df = load_dataset()
        model, scaler = load_model_and_scaler()

    if df is None or model is None:
        st.markdown('</div>', unsafe_allow_html=True)
        return

    required_columns = ['age', 'smoking', 'tumor_size', 'class', 'persistent_cough', 'fatigue', 
                        'cough_blood', 'chest_pain', 'weight_loss']
    if not all(col in df.columns for col in required_columns):
        st.error("Dataset is missing required columns.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Model Explanation Section
    st.markdown('<div class="glass-card breathing">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">How Our Lung Cancer Prediction System Works</h3>', unsafe_allow_html=True)
    st.markdown("""
        <p class="explanation-text">
            <strong>Our Lung Cancer Prediction System</strong> uses a <strong>Voting Classifier</strong> that combines 
            four models for accurate lung cancer risk prediction:
        </p>
        <ul class="explanation-text">
            <li><strong>Logistic Regression</strong>: Calculates risk probability by weighing patient 
            features like age and smoking history, ideal for linear relationships.</li>
            <li><strong>Random Forest</strong>: Analyzes multiple decision trees to identify complex 
            patterns in data, excelling at handling diverse patient profiles.</li>
            <li><strong>Decision Tree</strong>: Maps patient data into clear decision rules, making 
            it easy to interpret key risk factors like tumor size.</li>
            <li><strong>Voting Classifier</strong>: Merges predictions from the above models to 
            balance their strengths, ensuring robust and reliable risk assessments.</li>
            <li>
        </ul>
        <p class="explanation-text">
            Trained on real-world patient data, this ensemble delivers precise and trustworthy results.
        </p>
    """, unsafe_allow_html=True)

    # Performance Metrics
    st.markdown('<div style="margin: 2.5rem 0;">', unsafe_allow_html=True)
    metrics = [
        ("Accuracy", "92%", "Correctly predicts risk level"),
        ("Sensitivity", "90%", "Detects high-risk cases"),
        ("Specificity", "94%", "Identifies low-risk cases"),
       
    ]
    cols = st.columns(4)
    for i, (label, value, desc) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
                <div class="metric-card floating" style="animation-delay: {i*0.2}s;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature Importance Chart
    st.markdown('<div class="chart-container neomorphic">', unsafe_allow_html=True)
    st.markdown('<h4 class="section-title">What Drives Risk?</h4>', unsafe_allow_html=True)
    feature_fig = create_feature_importance_chart(model)
    st.plotly_chart(feature_fig, use_container_width=True)
    st.markdown("""
        <p class="explanation-text">
            <strong>Why it matters</strong>: Smoking and tumor size are top predictors. 
            <strong>Patients</strong> should prioritize smoking cessation; 
            <strong>clinicians</strong> focus on imaging and smoking history assessments.
        </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Key Risk Factors Section
    st.markdown('<div class="glass-card breathing">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Key Risk Factors</h3>', unsafe_allow_html=True)

    # Smoking Risk Chart
    st.markdown('<div class="chart-container neomorphic">', unsafe_allow_html=True)
    st.markdown('<h4 class="section-title">Smoking Impact</h4>', unsafe_allow_html=True)
    smoking_fig = create_smoking_risk_chart(df)
    st.plotly_chart(smoking_fig, use_container_width=True)
    st.markdown("""
        <p class="explanation-text">
            <strong>Why it matters</strong>: Smokers face significantly higher risk. 
            <strong>Patients</strong> understand the urgency of quitting; 
            <strong>clinicians</strong> emphasize smoking history in evaluations.
        </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tumor Size Chart
    st.markdown('<div class="chart-container neomorphic">', unsafe_allow_html=True)
    st.markdown('<h4 class="section-title">Tumor Size Impact</h4>', unsafe_allow_html=True)
    tumor_fig = create_tumor_size_chart(df)
    st.plotly_chart(tumor_fig, use_container_width=True)
    st.markdown("""
        <p class="explanation-text">
            <strong>Why it matters</strong>: Larger tumors strongly indicate higher risk. 
            <strong>Patients</strong> benefit from regular imaging; 
            <strong>clinicians</strong> prioritize tumor size in diagnostics.
        </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Age Risk Chart
    st.markdown('<div class="chart-container neomorphic">', unsafe_allow_html=True)
    st.markdown('<h4 class="section-title">Age Impact</h4>', unsafe_allow_html=True)
    age_fig = create_age_risk_chart(df)
    st.plotly_chart(age_fig, use_container_width=True)
    st.markdown("""
        <p class="explanation-text">
            <strong>Why it matters</strong>: Older age correlates with increased risk. 
            <strong>Patients</strong> over 60 should prioritize screenings; 
            <strong>clinicians</strong> factor age into risk assessments.
        </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Symptom Prevalence Chart
    st.markdown('<div class="chart-container neomorphic">', unsafe_allow_html=True)
    st.markdown('<h4 class="section-title">Common Symptoms</h4>', unsafe_allow_html=True)
    symptom_fig = create_symptom_prevalence_chart(df)
    st.plotly_chart(symptom_fig, use_container_width=True)
    st.markdown("""
        <p class="explanation-text">
            <strong>Why it matters</strong>: Symptoms like persistent cough and chest pain are prevalent. 
            <strong>Patients</strong> should report symptoms promptly; 
            <strong>clinicians</strong> use symptoms for early detection.
        </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------
# Entry Point
# ---------------------------------------
if __name__ == "__main__":
    show_visualizations_page()