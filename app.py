import streamlit as st
import pandas as pd
import os
import sqlite3
import bcrypt
from PIL import Image
import logging
from typing import Optional, Tuple

# Set page config FIRST
st.set_page_config(
    page_title="PulmoPredict AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "usersdata.db")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

# Refined color scheme
PRIMARY_COLOR = "#0284c7"
SECONDARY_COLOR = "#f8fafc"
ACCENT_COLOR = "#0369a1"
TEXT_COLOR = "#0f172a"
GRADIENT_START = "#0ea5e9"
GRADIENT_END = "#2563eb"
SUCCESS_COLOR = "#22c55e"
DANGER_COLOR = "#ef4444"

# Custom CSS
CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    body {{
        background-color: {SECONDARY_COLOR};
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR};
        line-height: 1.6;
    }}
    
    /* Hero section */
    .hero-section {{
        background: linear-gradient(145deg, {GRADIENT_START} 0%, {GRADIENT_END} 100%);
        border-radius: 24px;
        padding: 5rem 3rem;
        margin-bottom: 4rem;
        color: {SECONDARY_COLOR};
        text-align: center;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
        position: relative;
        overflow: hidden;
    }}
    
    .hero-section::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'/%3E%3C/svg%3E"),
        radial-gradient(circle at 10% 20%, rgba(255, 255, 255, 0.1) 0%, transparent 40%);
        opacity: 0.7;
    }}
    
    .hero-heading {{
        font-family: 'Poppins', sans-serif;
        font-size: 3.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 1.5rem;
        background: linear-gradient(to right, #ffffff, #e0f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    }}
    
    .hero-subheading {{
        font-size: 1.4rem;
        font-weight: 400;
        opacity: 0.95;
        margin-bottom: 2.75rem;
        max-width: 750px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.65;
    }}
    
    /* Buttons */
    .cta-button {{
        background: rgba(255, 255, 255, 0.94);
        color: {ACCENT_COLOR};
        padding: 1rem 2.4rem;
        border-radius: 14px;
        font-weight: 600;
        font-size: 1.05rem;
        text-decoration: none;
        border: none;
        margin: 0.7rem;
        display: inline-block;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }}
    
    .cta-button:hover {{
        background: {SECONDARY_COLOR};
        color: {GRADIENT_START};
        box-shadow: 0 7px 20px rgba(6, 182, 212, 0.25);
        transform: translateY(-3px);
    }}
    
    .cta-button-primary {{
        background: linear-gradient(135deg, {GRADIENT_START}, {GRADIENT_END});
        color: white;
        font-size: 1.15rem;
        padding: 1.1rem 2.75rem;
        border-radius: 14px;
        font-weight: 600;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(6, 182, 212, 0.2);
    }}
    
    .cta-button-primary:hover {{
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(6, 182, 212, 0.3);
    }}
    
    /* Section titles */
    .section-title {{
        text-align: center;
        color: {TEXT_COLOR};
        font-family: 'Poppins', sans-serif;
        font-size: 2.3rem;
        font-weight: 700;
        margin: 3.5rem 0 2.75rem;
        position: relative;
        display: inline-block;
        padding-bottom: 12px;
        letter-spacing: -0.01em;
    }}
    
    .section-title::after {{
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(to right, {GRADIENT_START}, {GRADIENT_END});
        border-radius: 10px;
    }}
    
    /* Stats bar */
    .stats-bar {{
        display: flex;
        justify-content: space-between;
        background: linear-gradient(100deg, {GRADIENT_START} 0%, {GRADIENT_END} 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin: 4.5rem 0;
        color: white;
        box-shadow: 0 10px 25px rgba(6, 182, 212, 0.15);
        position: relative;
        overflow: hidden;
    }}
    
    .stats-bar::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.08'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }}
    
    .stat-item {{
        text-align: center;
        padding: 0 1.5rem;
    }}
    
    .stat-value {{
        font-family: 'Poppins', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }}
    
    .stat-label {{
        font-size: 1rem;
        opacity: 0.95;
        letter-spacing: 0.05em;
    }}
    
    /* How it works */
    .step-container {{
        display: flex;
        justify-content: space-between;
        text-align: center;
        margin: 3.5rem 0;
        flex-wrap: wrap;
        gap: 25px;
    }}
    
    .step-item {{
        flex: 1;
        padding: 0 1.5rem;
        min-width: 220px;
        transition: transform 0.3s ease;
    }}
    
    .step-item:hover {{
        transform: translateY(-5px);
    }}
    
    .step-number {{
        background: linear-gradient(135deg, {GRADIENT_START}, {GRADIENT_END});
        width: 70px;
        height: 70px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.25rem;
        color: white;
        font-weight: bold;
        font-size: 1.75rem;
        box-shadow: 0 10px 20px rgba(6, 182, 212, 0.15);
    }}
    
    .step-title {{
        font-family: 'Poppins', sans-serif;
        color: {TEXT_COLOR};
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }}
    
    .step-description {{
        color: #475569;
        font-size: 1rem;
        line-height: 1.6;
    }}
    
    /* CTA section */
    .cta-section {{
        text-align: center;
        margin: 6rem 0;
        padding: 4rem 3rem;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.07), rgba(59, 130, 246, 0.07));
        border-radius: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(14, 165, 233, 0.1);
    }}
    
    .cta-section::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%230ea5e9' fill-opacity='0.05' fill-rule='evenodd'%3E%3Ccircle cx='3' cy='3' r='3'/%3E%3Ccircle cx='13' cy='13' r='3'/%3E%3C/g%3E%3C/svg%3E");
    }}
    
    .cta-title {{
        font-family: 'Poppins', sans-serif;
        color: {TEXT_COLOR};
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1.25rem;
        letter-spacing: -0.01em;
    }}
    
    .cta-description {{
        color: #475569;
        font-size: 1.15rem;
        margin-bottom: 2.5rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }}
    
    /* Footer */
    .footer {{
        margin-top: 6rem;
        padding: 3rem 0 2rem;
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
        border-top: 1px solid rgba(6, 182, 212, 0.08);
        background-color: rgba(248, 250, 252, 0.8);
    }}
    
    .footer-copyright {{
        margin-bottom: 0.6rem;
    }}
    
    .footer-tagline {{
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #94a3b8;
    }}
    
    /* Auth card */
    .auth-card {{
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        max-width: 400px;
        margin: auto;
        text-align: center;
    }}
    
    .auth-title {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.8rem;
        color: {TEXT_COLOR};
        margin-bottom: 1.5rem;
    }}
    
    /* Main container */
    .main-container {{
        max-width: 1000px;
        margin: auto;
        padding: 20px;
        background-color: {SECONDARY_COLOR};
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    /* Sidebar */
    .sidebar-content {{
        padding: 20px;
    }}
    
    .sidebar-title {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        color: {TEXT_COLOR};
        margin-bottom: 20px;
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.5); }}
        70% {{ box-shadow: 0 0 0 12px rgba(6, 182, 212, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); }}
    }}
    
    .animate-fade-in {{
        animation: fadeIn 0.9s ease-out forwards;
        opacity: 0;
    }}
    
    .animate-fade-in-delay-1 {{
        animation: fadeIn 0.9s ease-out 0.3s forwards;
        opacity: 0;
    }}
    
    .animate-fade-in-delay-2 {{
        animation: fadeIn 0.9s ease-out 0.6s forwards;
        opacity: 0;
    }}
    
    .pulse {{
        animation: pulse 2s infinite;
    }}
    
    /* Logo container */
    .logo-container {{
        padding: 1.25rem 0;
        display: flex;
        align-items: center;
    }}
    
    .logo-container img {{
        transition: transform 0.3s ease;
    }}
    
    .logo-container:hover img {{
        transform: scale(1.05);
    }}
    
    /* Responsive */
    @media (max-width: 768px) {{
        .hero-section {{
            padding: 4rem 1.5rem;
        }}
        .hero-heading {{
            font-size: 2.75rem;
        }}
        .hero-subheading {{
            font-size: 1.1rem;
        }}
        .stats-bar {{
            flex-direction: column;
            gap: 2rem;
        }}
        .stat-item {{
            padding: 1rem 0;
        }}
        .cta-section {{
            padding: 3rem 1.5rem;
        }}
        .cta-title {{
            font-size: 1.8rem;
        }}
    }}
</style>
"""

# Apply CSS after set_page_config
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Helper Functions
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def verify_password(password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed)
    except ValueError:
        return False

def create_usertable():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS userstable(username TEXT PRIMARY KEY, password BLOB)")
            conn.commit()
    except Exception as e:
        logging.error(f"Error creating user table: {e}")
        st.error("Database error. Please try again later.")

def login_user(username: str, password: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM userstable WHERE username = ?", (username,))
            result = c.fetchone()
        return verify_password(password, result[0]) if result else False
    except Exception as e:
        logging.error(f"Error logging in: {e}")
        return False

def add_userdata(username: str, password: str) -> Tuple[bool, str]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            hashed_pwd = hash_password(password)
            c.execute("INSERT INTO userstable (username, password) VALUES (?, ?)", (username, hashed_pwd))
            conn.commit()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        logging.error(f"Error adding user: {e}")
        return False, "An error occurred. Please try again."

def load_image(img_path: str) -> Optional[Image.Image]:
    try:
        return Image.open(img_path)
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        st.error(f"Error loading image: {e}")
        return None

# UI Components
def render_login_form():
    st.markdown('<div class="auth-card fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="auth-title">Welcome Back</h2>', unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_button = st.form_submit_button("Login", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    return username, password, login_button

def render_signup_form():
    st.markdown('<div class="auth-card fade-in">', unsafe_allow_html=True)
    st.markdown('<h2 class="auth-title">Join PulmoPredict AI</h2>', unsafe_allow_html=True)
    with st.form("signup_form"):
        new_username = st.text_input("Username", placeholder="Choose a username")
        new_password = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
        signup_button = st.form_submit_button("Create Account", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    return new_username, new_password, confirm_password, signup_button

def show_home_page():
    st.markdown('<div style="max-width: 1250px; margin: 0 auto; padding: 0 25px;">', unsafe_allow_html=True)
    
    # Logo
    if os.path.exists(LOGO_PATH):
        col1, col2, col3 = st.columns([1, 10, 1])
        with col1:
            st.markdown('<div class="logo-container">', unsafe_allow_html=True)
            st.image(load_image(LOGO_PATH), width=120)
            st.markdown('</div>', unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-section animate-fade-in">
        <h1 class="hero-heading">Lung Cancer Prediction App</h1>
        <p class="hero-subheading">Harnessing artificial intelligence to revolutionize early lung cancer detection, empowering you with clear, actionable insights for improved health outcomes.</p>
        <div>
            <a href="?page=Login" class="cta-button pulse">Login</a>
            <a href="?page=Signup" class="cta-button">Sign Up</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats Section
    st.markdown("""
    <div class="stats-bar animate-fade-in-delay-1">
        <div class="stat-item">
            <div class="stat-value">97%</div>
            <div class="stat-label">Accuracy Rate</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">10x</div>
            <div class="stat-label">Faster Analysis</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">24/7</div>
            <div class="stat-label">Availability</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">3+</div>
            <div class="stat-label">Combined Algorithms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("""
    <div class="animate-fade-in-delay-2" style="margin: 5rem 0; text-align: center;">
        <h2 class="section-title">How It Works</h2>
        <div class="step-container">
            <div class="step-item">
                <div class="step-number">1</div>
                <h3 class="step-title">Enter Your Data</h3>
                <p class="step-description">Provide your health information through our secure, intuitive interface.</p>
            </div>
            <div class="step-item">
                <div class="step-number">2</div>
                <h3 class="step-title">AI Analysis</h3>
                <p class="step-description">Our advanced algorithms process your data with clinical precision.</p>
            </div>
            <div class="step-item">
                <div class="step-number">3</div>
                <h3 class="step-title">Comprehensive Results</h3>
                <p class="step-description">Receive detailed insights through elegant visualizations and actionable guidance.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Call to Action
    st.markdown("""
    <div class="cta-section">
        <h2 class="cta-title">Take Charge of Your Lung Health</h2>
        <p class="cta-description">Join thousands of healthcare professionals and patients using PulmoPredict AI to stay ahead with early detection and proactive care.</p>
        <a href="?page=Signup" class="cta-button-primary">Get Started Now</a>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p class="footer-copyright">© 2025 Lung Cancer Prediction App. All rights reserved.</p>
        <p class="footer-tagline">Transforming lung health through cutting-edge AI innovation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# App Class
class PulmoPredictApp:
    def __init__(self):
        # Initialize session state
        self.init_session_state()

    def init_session_state(self):
        if "user_authenticated" not in st.session_state:
            st.session_state.user_authenticated = False
        if "username" not in st.session_state:
            st.session_state.username = ""
        if "page" not in st.session_state:
            st.session_state.page = "Home"

    def show_login(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username, password, login_button = render_login_form()
            if login_button:
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    create_usertable()
                    if login_user(username, password):
                        st.session_state.user_authenticated = True
                        st.session_state.username = username
                        st.session_state.page = "App"
                        st.success(f"Welcome, {username}!")
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")

    def show_signup(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            new_username, new_password, confirm_password, signup_button = render_signup_form()
            if signup_button:
                if not new_username or not new_password or not confirm_password:
                    st.error("Please fill in all fields.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    create_usertable()
                    success, message = add_userdata(new_username, new_password)
                    if success:
                        st.success(message + " Please log in.")
                        st.session_state.page = "Login"
                        st.rerun()
                    else:
                        st.error(message)

    def show_app(self):
        # Sidebar
        with st.sidebar:
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            if os.path.exists(LOGO_PATH):
                st.image(load_image(LOGO_PATH), width=120)
            st.markdown(f'<h3 class="sidebar-title">Welcome, {st.session_state.username}</h3>', unsafe_allow_html=True)
            if st.button("Logout", key="logout", use_container_width=True):
                st.session_state.user_authenticated = False
                st.session_state.username = ""
                st.session_state.page = "Home"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Main Content
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def run(self):
        if not st.session_state.user_authenticated:
            if st.query_params.get("page"):
                st.session_state.page = st.query_params["page"]
            if st.session_state.page == "Home":
                show_home_page()
            elif st.session_state.page == "Login":
                self.show_login()
            elif st.session_state.page == "Signup":
                self.show_signup()
        else:
            self.show_app()

if __name__ == "__main__":
    app = PulmoPredictApp()
    app.run()