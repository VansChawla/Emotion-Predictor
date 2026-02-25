import streamlit as st

st.set_page_config(
    page_title="Emotion AI",
    page_icon="happy-face.png",
    layout="centered",
    initial_sidebar_state="auto" # "auto", "expanded", or "collapsed"
)

import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 1. DOWNLOAD NLTK RESOURCES (CACHED) ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

# --- 2. CSS STYLING (RESPONSIVE) ---
def local_css():
    st.markdown("""
    <style>
    .block-container {
        padding-top: 3rem;
        padding-bottom: 5rem;
    }
    
    .stApp {
        /* background-color: #f0f2f6; (Uncomment if needed) */
    }

    h1 {
        color: #ff4b4b; /* Your Original Color */
        font-family: 'Helvetica', sans-serif;
        text-align: center;
    }

    .stTextArea textarea {
        border-radius: 10px;
    }

    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 8px 20px;
        font-size: 16px;
        transition: 0.3s;
        display: block;
        margin: 0 auto;
        white-space: nowrap;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }

    /* Hover Effect */
    .stButton > button:hover {
        background-color: #ff1a1a; /* Your Original Hover Color */
        color: white;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* --- ADDED RESPONSIVE LOGIC --- */

    /* DESKTOP (Screens larger than 600px) */
    @media only screen and (min-width: 600px) {
        .stButton > button {
            width: 100%; /* On PC, button is 50% width (looks cleaner) */
        }
    }

    /* MOBILE (Screens smaller than 600px) */
    @media only screen and (max-width: 600px) {
        .stButton > button {
            width: 100%;
        }
        
        h1 {
            font-size: 24px; /* Smaller title on mobile */
        }
        
        .block-container {
            padding-top: 1rem; /* Less empty space at the top on mobile */
        }
    }

    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. LOAD SAVED ASSETS ---
def load_assets():
    try:
        with open('logistic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        mapping = {
            0: 'sadness', 
            1: 'anger', 
            2: 'love', 
            3: 'surprise', 
            4: 'fear', 
            5: 'joy'
        }
        return model, vectorizer, mapping
    except FileNotFoundError:
        st.error("⚠️ Error: Model files not found. Please ensure logistic_model.pkl, tfidf_vectorizer.pkl, and label_mapping.pkl are in the same folder as this script.")
        return None, None, None

model, vectorizer, label_mapping = load_assets()

# --- 4. PREPROCESSING FUNCTION ---
def preprocess_text(text):
    # A. Lowercase
    text = text.lower()
    
    # B. Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # C. Remove Numbers
    text = "".join([i for i in text if not i.isdigit()])
    
    # D. Remove Emojis (Basic ASCII check)
    text = "".join([i for i in text if i.isascii()])
    
    # E. Tokenize & Remove Stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    cleaned_words = [i for i in words if i not in stop_words]
    
    return ' '.join(cleaned_words)

# --- 5. STREAMLIT USER INTERFACE ---
st.title("Emotion Detection from Text")
st.markdown('<p style="text-align: center; margin-top: -20px;">(Enter a sentence below, and the AI will predict the underlying emotion)</p>', unsafe_allow_html=True)

# Text Input Area
user_input = st.text_area("Type your text here:", height=150, placeholder="e.g., I feel so happy today because I passed my exam!")

# Prediction Logic
if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    elif model is not None:
        # Step A: Preprocess the input
        clean_text = preprocess_text(user_input)
        
        # Step B: Vectorize
        # Use .transform() because the vectorizer is already fitted
        vectorized_text = vectorizer.transform([clean_text])
        
        # Step C: Predict
        prediction_index = model.predict(vectorized_text)[0]
        prediction_label = label_mapping.get(prediction_index, "Unknown")

        # st.write(f"Debug Info: Model predicted Number **{prediction_index}**")
        
        # Step D: Display Result
        st.success(f"Predicted Emotion: **{prediction_label.upper()}**")
        
        # Optional: Display Confidence Scores
        try:
            probs = model.predict_proba(vectorized_text)[0]
            st.write("---")
            st.write("### Confidence Scores:")
            
            # Create columns for a nicer layout
            for idx, score in enumerate(probs):
                label = label_mapping.get(idx, str(idx)).capitalize()
                st.progress(score, text=f"{label}: {score*100:.1f}%")
        except AttributeError:
            st.info("Probability scores not available for this model.")