# correlation_app.py
import streamlit as st

# Set page config and style
st.set_page_config(page_title="Welcome",page_icon="📊", layout="wide")
hide_st_style = """
    <style>
    footer {visibility: hidden;}
    .stSelectbox, .stButton, .stTextInput {font-size: 16px;}
    .stMarkdown {padding-top: 0.8rem; padding-bottom: 1.2rem;}
    .css-ffhzg2 {background-color: #f0f4f8;}
    .css-1d391kg {font-family: 'Roboto', sans-serif;}
    .stText {font-size: 16px;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- Welcome Page ---
def show_welcome_page():
    # Title section
    st.markdown("<h1 style='text-align: center;'>Welcome to TATA Commercial CSI Analysis 2024/2025</h1>", unsafe_allow_html=True)
    
    # Description section
    st.markdown("""
    <div style='text-align: center; font-size: 20px;'>
        <p>📊 Explore detailed correlation analysis between various service attributes and overall satisfaction scores.</p>
        <p>The insights gained will help enhance customer experiences and optimize performance in various branches.</p>
        <p>🔍 Select a branch to begin analyzing correlations between service advisors and different performance factors.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Main Flow ---
show_welcome_page()

# Add copyright at the bottom
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #071429;
        color: #b4b8bf;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        © 2025 DIMO Customer Experience. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)