import streamlit as st
import joblib
import pandas as pd
import string

# 1. Page Config (Set to dark mode natively)
st.set_page_config(page_title="AI Detector // SYS.TERMINAL", page_icon="🤖", layout="centered")

# 2. Cyberpunk CSS Injection
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #050510;
        color: #00ffcc;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Headers with neon pink glow */
    h1, h2, h3 {
        color: #ff0055 !important;
        text-shadow: 0 0 10px #ff0055, 0 0 20px #ff0055;
        text-transform: uppercase;
    }

    /* Text area with cyan glow and yellow text */
    .stTextArea textarea {
        background-color: #0a0a1a !important;
        color: #fcee0a !important;
        border: 1px solid #00ffcc !important;
        box-shadow: 0 0 10px #00ffcc !important;
        font-family: 'Courier New', Courier, monospace !important;
    }

    /* Scan Button styling */
    .stButton>button {
        background-color: #050510 !important;
        color: #00ffcc !important;
        border: 2px solid #00ffcc !important;
        box-shadow: 0 0 10px #00ffcc !important;
        font-weight: bold;
        text-transform: uppercase;
        font-family: 'Courier New', Courier, monospace !important;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #00ffcc !important;
        color: #050510 !important;
        box-shadow: 0 0 20px #00ffcc !important;
    }

    /* Custom Output Boxes */
    .cyber-box-ai {
        background-color: #1a0510;
        border: 2px solid #ff0055;
        box-shadow: 0 0 15px #ff0055;
        padding: 20px;
        color: #ff0055;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
        text-transform: uppercase;
    }
    
    .cyber-box-human {
        background-color: #051a1a;
        border: 2px solid #00ffcc;
        box-shadow: 0 0 15px #00ffcc;
        padding: 20px;
        color: #00ffcc;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('ai_stylometric_pipeline.pkl')

pipeline = load_pipeline()

# 4. App UI
st.title("SYS.TERMINAL // AI_DETECTOR")
st.write("DEVELOPER: ALI ALSSANM // INITIATING STYLOMETRIC ANALYSIS...")

user_input = st.text_area(">> INPUT_TEXT_DATA_FOR_ANALYSIS:", height=200)

if st.button("EXECUTE_SCAN"):
    if user_input.strip() == "":
        st.warning(">> ERROR: NO_DATA_DETECTED. PLEASE INPUT TEXT.")
    else:
        # Calculate features
        char_count = len(user_input)
        word_count = len(user_input.split())
        avg_word_len = char_count / (word_count + 1)
        punct_count = sum([1 for char in user_input if char in string.punctuation])
        
        # Package into DataFrame
        input_df = pd.DataFrame([{
            'text': user_input,
            'char_count': char_count,
            'word_count': word_count,
            'avg_word_len': avg_word_len,
            'punct_count': punct_count
        }])

        # Predict
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        st.markdown("### >> SCAN_COMPLETE")
        
        # Display custom glowing HTML boxes instead of standard Streamlit alerts
        if prediction == 1:
            st.markdown(f'<div class="cyber-box-ai">⚠️ WARNING: SYNTHETIC SIGNATURE DETECTED<br><br>CONFIDENCE: {probability:.1%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="cyber-box-human">🟢 HUMAN AUTHORSHIP VERIFIED<br><br>CONFIDENCE: {(1 - probability):.1%}</div>', unsafe_allow_html=True)