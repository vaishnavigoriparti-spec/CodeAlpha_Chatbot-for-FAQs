import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import re
import html
import ssl
from datetime import datetime

# ==========================================
# 1. ROBUST NLTK DOWNLOADS & CONFIG
# ==========================================
# Disable SSL verification for NLTK downlods to prevent Windows certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Initialize session state for loading NLTK resources
if "nltk_downloaded" not in st.session_state:
    st.session_state.nltk_downloaded = False

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_resources():
    """Silently download all required NLTK resources with status checks."""
    if st.session_state.nltk_downloaded:
        return
    
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            st.error(f"Failed to download NLTK '{res}': {e}")
    st.session_state.nltk_downloaded = True

# ==========================================
# 2. STREAMLIT PAGE INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Call NLTK downloader with a beautiful loading animation
if not st.session_state.nltk_downloaded:
    with st.spinner("🧠 Initializing AI Core & NLP Models... Please wait..."):
        download_nltk_resources()

# ==========================================
# 3. GLOBAL TEXT PREPROCESSOR & ALGORITHMS
# ==========================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception:
    lemmatizer = None
    stop_words = None

def preprocess_text(text):
    """Tokenize, clean, remove stopwords, and lemmatize text."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase & strip
    text = text.lower().strip()
    
    # Remove punctuation & special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Fallback to simple split if NLTK is failed/not loaded
    if not lemmatizer or not stop_words:
        return " ".join([word for word in text.split() if len(word) > 1])
    
    try:
        tokens = word_tokenize(text)
        # Filter out stopwords and lemmatize
        cleaned_tokens = [
            lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in stop_words and len(token) > 0
        ]
        return " ".join(cleaned_tokens)
    except Exception:
        # Secondary fallback
        return " ".join([word for word in text.split() if len(word) > 1])

# ==========================================
# 4. LOAD FAQ DATASET WITH FALLBACKS
# ==========================================
@st.cache_data
def load_faq_data():
    """Load FAQ dataset from CSV. Falls back to pre-defined dataset if file error."""
    csv_path = "faq.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Ensure proper headers exist
            if 'Question' in df.columns and 'Answer' in df.columns:
                return df
        except Exception as e:
            st.error(f"Error loading {csv_path}: {e}")
            
    # Premium fallback FAQ dataset in case CSV is missing or broken
    fallback_data = {
        'Question': [
            "What is Artificial Intelligence (AI)?",
            "What is Machine Learning (ML)?",
            "What is Deep Learning (DL)?",
            "What is Natural Language Processing (NLP)?",
            "What is the difference between supervised and unsupervised learning?",
            "What is reinforcement learning?",
            "What is overfitting in machine learning?",
            "How can we prevent overfitting?",
            "What is underfitting?",
            "What is TF-IDF?",
            "What is Cosine Similarity?",
            "What is the purpose of this AI FAQ Chatbot?",
            "What is the core matching algorithm used here?",
            "What are some common applications of NLP?"
        ],
        'Answer': [
            "AI is a branch of computer science that focuses on building smart machines capable of performing tasks that typically require human intelligence, such as reasoning, learning, decision-making, and natural language understanding.",
            "Machine Learning is a subset of AI that allows systems to automatically learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can analyze data and make predictions.",
            "Deep Learning is a subset of Machine Learning that uses multi-layered artificial neural networks (inspired by the human brain) to model and solve complex patterns in large amounts of unstructured data like images, audio, and text.",
            "Natural Language Processing is a field of AI that enables computers to understand, interpret, manipulate, and generate human language. Examples include machine translation, sentiment analysis, and chatbots.",
            "Supervised learning uses labeled training data (data with known output labels) to train models, while unsupervised learning analyzes unlabeled data to find hidden patterns, clusters, or structures on its own.",
            "Reinforcement Learning (RL) is an ML paradigm where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. It is based on a trial-and-error feedback loop.",
            "Overfitting occurs when a machine learning model learns the noise and details of the training data too well, resulting in excellent performance on training data but poor generalization to new, unseen data.",
            "Overfitting can be prevented by using techniques like regularization, cross-validation, simplifying the model architecture, training with more data, and using dropout layers in neural networks.",
            "Underfitting occurs when a model is too simple to capture the underlying pattern of the data, resulting in poor performance on both the training data and new test data.",
            "TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a statistical numerical metric that reflects how important a word is to a document in a collection or corpus, commonly used in text matching.",
            "Cosine Similarity is a metric used to measure how similar two vectors are by calculating the cosine of the angle between them in a multi-dimensional space. In text matching, it determines similarity between TF-IDF document vectors regardless of size.",
            "This chatbot is designed to provide quick, accurate, and contextually relevant answers to common questions about Artificial Intelligence, Machine Learning, Deep Learning, and Natural Language Processing.",
            "The chatbot utilizes a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer to index the textual questions, and evaluates incoming queries using a mathematical Cosine Similarity metric to identify the closest match.",
            "Natural Language Processing (NLP) is used in a wide range of real-world applications, including virtual assistants, machine translation (e.g. Google Translate), sentiment analysis, text summarization, spam filters, and conversational chatbots."
        ]
    }
    return pd.DataFrame(fallback_data)

faq_df = load_faq_data()

# ==========================================
# 5. CUSTOM STYLING (GLASSMORPHISM VIBE)
# ==========================================
def apply_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Modern Banner Design */
    .header-container {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.15) 0%, rgba(255, 101, 132, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        text-align: center;
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #6c63ff 0%, #ff6584 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 4px 10px rgba(108, 99, 255, 0.15);
    }
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        color: #7d8597;
        margin-top: 10px;
        margin-bottom: 0;
    }
    
    /* Info Card UI */
    .info-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    .info-card:hover {
        transform: translateY(-3px);
        border-color: rgba(108, 99, 255, 0.2);
    }
    
    /* Metrics Badge */
    .metric-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6c63ff, #8b5cf6);
        color: white;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .metric-badge-low {
        display: inline-block;
        background: linear-gradient(135deg, #ff6584, #f43f5e);
        color: white;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
    }
    
    /* Quick Actions */
    .quick-title {
        font-weight: 600;
        font-size: 1rem;
        color: #6c63ff;
        margin-bottom: 10px;
    }
    
    /* Custom Sidebar Avatar Header */
    .sidebar-profile {
        text-align: center;
        padding: 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 20px;
    }
    .sidebar-avatar {
        font-size: 4rem;
        margin-bottom: 10px;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    /* Chat Avatars & Area Styling */
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        font-size: 0.98rem;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# ==========================================
# 6. IN-LINE HTML TEXT-TO-SPEECH HELPER
# ==========================================
def get_tts_button_html(text, message_id):
    """Generate professional, zero-dependency, safe Web Speech API button."""
    escaped_text = html.escape(text.replace("'", "\\'").replace("\n", " "))
    button_id = f"tts_btn_{message_id}"
    
    html_content = f"""
    <div style="display: flex; justify-content: flex-end; margin-top: -10px; margin-bottom: 15px;">
        <button id="{button_id}" class="custom-tts-btn" onclick="
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance('{escaped_text}');
            msg.rate = 1.0;
            msg.pitch = 1.05;
            // Find an English voice if available
            var voices = window.speechSynthesis.getVoices();
            var englishVoice = voices.find(v => v.lang.startsWith('en'));
            if(englishVoice) msg.voice = englishVoice;
            
            window.speechSynthesis.speak(msg);
            var btn = document.getElementById('{button_id}');
            btn.innerHTML = '🔊 Speaking...';
            msg.onend = function() {{ btn.innerHTML = '🔊 Listen'; }};
        ">🔊 Listen</button>
    </div>
    <style>
        .custom-tts-btn {{
            background: linear-gradient(135deg, rgba(108, 99, 255, 0.08), rgba(255, 101, 132, 0.08));
            color: #6c63ff;
            border: 1px solid rgba(108, 99, 255, 0.25);
            border-radius: 10px;
            padding: 4px 12px;
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            font-weight: 500;
            outline: none;
        }}
        .custom-tts-btn:hover {{
            background: linear-gradient(135deg, #6c63ff, #ff6584);
            color: white !important;
            border-color: transparent;
            box-shadow: 0 4px 12px rgba(108, 99, 255, 0.25);
            transform: translateY(-1px);
        }}
        .custom-tts-btn:active {{
            transform: translateY(0px);
        }}
    </style>
    """
    return html_content

# ==========================================
# 7. CHATBOT MATHS & TF-IDF CORE ENGINE
# ==========================================
def calculate_best_faq_match(query, faq_data, threshold=0.30):
    """Perform text cleaning, TF-IDF vectorization, and cosine similarity matching."""
    # Preprocess questions
    faq_data['Processed_Question'] = faq_data['Question'].apply(preprocess_text)
    
    # Preprocess user query
    processed_query = preprocess_text(query)
    
    # Fallback to raw lowercase if the clean result is empty (e.g. stopword-only queries)
    if not processed_query.strip():
        processed_query = query.lower().strip()
        
    try:
        # Initialize Scikit-Learn TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        tfidf_matrix = vectorizer.fit_transform(faq_data['Processed_Question'])
        
        # Transform user query
        query_vector = vectorizer.transform([processed_query])
        
        # Calculate cosine similarities between query and each FAQ question
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get maximum similarity details
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        # Check against confidence threshold
        if best_score >= threshold:
            answer = faq_data.iloc[best_idx]['Answer']
            matched_question = faq_data.iloc[best_idx]['Question']
            return answer, best_score, matched_question
        else:
            return "Sorry, I couldn't understand your question. Could you try rephrasing it or choose one of the suggested FAQs below?", best_score, None
            
    except Exception as e:
        return f"⚠️ An algorithmic processing error occurred: {str(e)}", 0.0, None

# ==========================================
# 8. STATE MANAGEMENT & SYSTEM FLOW
# ==========================================
# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I am your intelligent **AI FAQ Assistant**. How can I help you understand AI, Machine Learning, Deep Learning, or NLP today? 🚀",
        "score": 1.0,
        "match": "Welcome greeting",
        "id": "welcome"
    })

# Handles suggested question clicks
if "suggestion_clicked" not in st.session_state:
    st.session_state.suggestion_clicked = None

def trigger_suggestion(question):
    st.session_state.suggestion_clicked = question

# ==========================================
# 9. GRAPHICAL USER INTERFACE
# ==========================================

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-profile">
        <div class="sidebar-avatar">🤖</div>
        <h2 style='margin: 0; font-size: 1.5rem; font-weight: 600;'>AI FAQ Bot</h2>
        <p style='margin: 5px 0 0 0; color: #6c63ff; font-size: 0.85rem; letter-spacing: 1px; font-weight: 600;'>INTELLIGENT ASSISTANT</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section 1: System Info
    st.markdown("### 📊 System Information")
    st.markdown("""
    <div class="info-card">
        <strong>Engine:</strong> TF-IDF Vectorizer<br/>
        <strong>Metric:</strong> Cosine Similarity<br/>
        <strong>Interface:</strong> Streamlit Wide Mode<br/>
        <strong>Backend:</strong> Scikit-Learn & NLTK<br/>
        <strong>Status:</strong> Active ✅
    </div>
    """, unsafe_allow_html=True)
    
    # Section 2: Algorithmic Controls
    st.markdown("### ⚙️ Engine Parameters")
    conf_threshold = st.slider(
        "Cosine Confidence Threshold 🎯", 
        min_value=0.10, 
        max_value=1.00, 
        value=0.30, 
        step=0.05,
        help="Higher values demand an exact phrasing match; lower values allow broader contextual similarities."
    )
    
    # Section 3: FAQ Dataset Viewer
    st.markdown("### 📚 Training FAQ Corpus")
    with st.expander("Expand FAQ Dataset", expanded=False):
        st.dataframe(
            faq_df[['Question']], 
            use_container_width=True, 
            hide_index=True
        )
        st.markdown(f"_Total loaded FAQs: **{len(faq_df)} rows**_")
        
    # Section 4: Utilities
    st.markdown("### 🛠️ Utilities")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.suggestion_clicked = None
            st.rerun()
    with col2:
        # Download Chat History logic
        if len(st.session_state.messages) > 1:
            chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
            st.download_button(
                label="📥 Export",
                data=chat_text,
                file_name="faq_chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.button("📥 Export", disabled=True, use_container_width=True)

# Main Screen Header Layout
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🤖 AI FAQ Intelligent Chatbot</h1>
    <p class="header-subtitle">Fully Powered by Advanced NLP, Vector Space Modeling & Cosine Similarity Algorithms</p>
</div>
""", unsafe_allow_html=True)

# Top stats banner
col_stat1, col_stat2, col_stat3 = st.columns(3)
with col_stat1:
    st.info(f"📊 **FAQ Database Size:** {len(faq_df)} preprocessed questions", icon="💾")
with col_stat2:
    st.success(f"⚡ **Matching Engine:** TF-IDF Bag-of-Words Model", icon="🔥")
with col_stat3:
    st.warning(f"🎯 **Min Match Confidence:** {int(conf_threshold*100)}%", icon="🛡️")

st.markdown("<br/>", unsafe_allow_html=True)

# Render Chat History
for idx, message in enumerate(st.session_state.messages):
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # Display engineering details if available for bot answers
        if message["role"] == "assistant" and "score" in message:
            score = message["score"]
            match_type = message["match"]
            
            # Show a metric badge based on threshold
            if score >= conf_threshold:
                st.markdown(
                    f"""<div class='metric-badge'>Confidence Score: {score:.2f} ({int(score*100)}%) • Match: {html.escape(str(match_type))}</div>""", 
                    unsafe_allow_html=True
                )
            elif score > 0.0:
                st.markdown(
                    f"""<div class='metric-badge-low'>Low Confidence: {score:.2f} ({int(score*100)}%) • Below Threshold ({int(conf_threshold*100)}%)</div>""", 
                    unsafe_allow_html=True
                )
            
            # Play inline Voice Speech Synthesizer if score is positive
            if score > 0.0 and message["id"] != "welcome":
                st.markdown(get_tts_button_html(message["content"], f"msg_{idx}"), unsafe_allow_html=True)

# Handle Suggestion Clicked Event
user_query = st.session_state.suggestion_clicked

# Capture Text Input
chat_input_val = st.chat_input("Ask a question (e.g., 'What is machine learning?' or 'how to prevent overfitting?')")
if chat_input_val:
    user_query = chat_input_val

# Processing User Query
if user_query:
    # Reset suggestion clicked state
    st.session_state.suggestion_clicked = None
    
    # 1. Print User Chat Bubble
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
        
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })
    
    # 2. Compute matching response
    with st.chat_message("assistant", avatar="🤖"):
        # Let's show a loading typewriter spinner
        with st.spinner("🔍 Scanning Vector Space for matches..."):
            ans, score, matched_q = calculate_best_faq_match(user_query, faq_df, conf_threshold)
            time.sleep(0.3) # Subtle latency for realism
            
        # Stream output using typewriter effect
        response_placeholder = st.empty()
        full_response = ""
        for word in ans.split(" "):
            full_response += word + " "
            response_placeholder.markdown(full_response + "▌")
            time.sleep(0.02)
        response_placeholder.markdown(full_response)
        
        # Display engineering details
        if score >= conf_threshold:
            st.markdown(
                f"""<div class='metric-badge'>Confidence Score: {score:.2f} ({int(score*100)}%) • Match: {html.escape(str(matched_q))}</div>""", 
                unsafe_allow_html=True
            )
        elif score > 0.0:
            st.markdown(
                f"""<div class='metric-badge-low'>Low Confidence: {score:.2f} ({int(score*100)}%) • Below Threshold ({int(conf_threshold*100)}%)</div>""", 
                unsafe_allow_html=True
            )
            
        # Render TTS button
        msg_id = f"msg_{len(st.session_state.messages)}"
        if score > 0.0:
            st.markdown(get_tts_button_html(ans, msg_id), unsafe_allow_html=True)
            
    # Append assistant's message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": ans,
        "score": float(score),
        "match": matched_q if matched_q else "None (Low Confidence)",
        "id": msg_id
    })
    st.rerun()

# Display Suggested FAQ Chips at bottom (if no current conversation processing)
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("""<div class="quick-title">💡 Frequently Asked Questions (Click to Ask)</div>""", unsafe_allow_html=True)
cols = st.columns(3)
sample_prompts = [
    "What is Machine Learning?",
    "What is Deep Learning?",
    "What is Cosine Similarity?",
    "What is tokenization?",
    "How can we prevent overfitting?",
    "What is the difference between stemming and lemmatization?"
]

for index, prompt in enumerate(sample_prompts):
    col_idx = index % 3
    with cols[col_idx]:
        if st.button(f"🔍 {prompt}", key=f"sug_{index}", use_container_width=True):
            trigger_suggestion(prompt)
            st.rerun()
