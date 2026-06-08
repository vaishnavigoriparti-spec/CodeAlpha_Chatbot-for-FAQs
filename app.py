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
# 1. DEPENDENCY SETUP & RESOURCE PROCURING
# ==========================================
# Suppress SSL verification warnings for NLTK package downloads on local machine
try:
    _ssl_context_override = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _ssl_context_override

# Track initialization status in session state
if "nlp_initialized" not in st.session_state:
    st.session_state.nlp_initialized = False

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def initialize_nlp_dependencies():
    """Ensure all required natural language processing models are downloaded locally."""
    if st.session_state.nlp_initialized:
        return
    
    required_packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as err:
            st.error(f"Dependency download error for '{package}': {err}")
    st.session_state.nlp_initialized = True

# ==========================================
# 2. APPLICATION INITIAL CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AuraFAQ - Intelligent Semantic Assistant",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show a premium glowing loader during NLTK loading
if not st.session_state.nlp_initialized:
    with st.spinner("🌌 Initializing AI Core & Semantic Processing Systems..."):
        initialize_nlp_dependencies()

# ==========================================
# 3. TEXT PREPROCESSING PIPELINE
# ==========================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    word_lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words('english'))
except Exception:
    word_lemmatizer = None
    english_stopwords = None

def clean_and_tokenize(input_text):
    """Normalize, strip punctuation, remove noise, and lemmatize textual tokens."""
    if not isinstance(input_text, str):
        return ""
    
    # Normalize text to lower case and strip whitespaces
    input_text = input_text.lower().strip()
    
    # Strip non-alphanumeric character structures
    input_text = re.sub(r'[^a-zA-Z0-9\s]', '', input_text)
    
    # Simple split fallback if NLTK tools failed to initialize
    if not word_lemmatizer or not english_stopwords:
        return " ".join([word for word in input_text.split() if len(word) > 1])
    
    try:
        token_list = word_tokenize(input_text)
        # Filter stopwords and lemmatize to base grammatical form
        refined_tokens = [
            word_lemmatizer.lemmatize(token) 
            for token in token_list 
            if token not in english_stopwords and len(token) > 0
        ]
        return " ".join(refined_tokens)
    except Exception:
        # Secondary basic fallback
        return " ".join([word for word in input_text.split() if len(word) > 1])

# ==========================================
# 4. CORPUS ACQUISITION (FAQ DATASET)
# ==========================================
@st.cache_data
def retrieve_faq_corpus():
    """Retrieve FAQ database from local storage, fallback to precompiled data if missing."""
    data_source = "faq.csv"
    if os.path.exists(data_source):
        try:
            dataframe = pd.read_csv(data_source)
            if 'Question' in dataframe.columns and 'Answer' in dataframe.columns:
                return dataframe
        except Exception as error:
            st.error(f"Error reading dataset files: {error}")
            
    # Default fallback FAQ dataset
    default_faq = {
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
    return pd.DataFrame(default_faq)

faq_dataframe = retrieve_faq_corpus()

# ==========================================
# 5. CYBER MIDNIGHT GLOWING THEME (CSS)
# ==========================================
def inject_cyber_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Fonts and Basic Overrides */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Cyber Midnight Header Banner */
    .dashboard-header {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 50%, rgba(102, 126, 234, 0.1) 100%);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 20px;
        padding: 35px 20px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -50%;
        width: 200%;
        height: 100%;
        background: radial-gradient(circle, rgba(0, 242, 254, 0.05) 0%, transparent 60%);
        pointer-events: none;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0px 2px 8px rgba(0, 242, 254, 0.3));
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-size: 1.15rem;
        font-weight: 400;
        color: #94a3b8;
        margin-top: 12px;
        margin-bottom: 0;
        letter-spacing: 0.2px;
    }
    
    /* Sleek Translucent Metric Sidebar Cards */
    .system-card {
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(79, 172, 254, 0.15);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .system-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.4);
        box-shadow: 0 8px 32px rgba(0, 242, 254, 0.1);
    }
    
    /* Premium Accuracy Badges */
    .accuracy-indicator-high {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.15), rgba(79, 172, 254, 0.15));
        border: 1px solid rgba(0, 242, 254, 0.4);
        color: #00f2fe;
        padding: 5px 14px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 10px;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.1);
    }
    
    .accuracy-indicator-low {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.12), rgba(225, 29, 72, 0.12));
        border: 1px solid rgba(244, 63, 94, 0.4);
        color: #fb7185;
        padding: 5px 14px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 10px;
        box-shadow: 0 0 10px rgba(244, 63, 94, 0.1);
    }
    
    /* Interactive Prompts Headers */
    .quick-prompts-header {
        font-weight: 600;
        font-size: 1.05rem;
        color: #00f2fe;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Avatar Pulsing Side Panel UI */
    .avatar-wrapper {
        text-align: center;
        padding: 25px 0 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 25px;
    }
    
    .pulsing-avatar {
        font-size: 4.2rem;
        margin-bottom: 12px;
        display: inline-block;
        position: relative;
        animation: pulseAvatar 4s ease-in-out infinite;
    }
    
    @keyframes pulseAvatar {
        0% { transform: translateY(0px) scale(1); filter: drop-shadow(0 0 0px rgba(0, 242, 254, 0)); }
        50% { transform: translateY(-8px) scale(1.03); filter: drop-shadow(0 4px 12px rgba(0, 242, 254, 0.35)); }
        100% { transform: translateY(0px) scale(1); filter: drop-shadow(0 0 0px rgba(0, 242, 254, 0)); }
    }
    
    /* Customizing Streamlit Native Chat Interfaces */
    [data-testid="stChatMessage"] {
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.03);
        background-color: rgba(30, 41, 59, 0.2) !important;
        backdrop-filter: blur(4px);
        transition: border 0.3s ease;
    }
    
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(0, 242, 254, 0.12);
    }
    
    /* Chat Bubble Markdown font adjustments */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        font-size: 1rem;
        line-height: 1.6;
        color: #e2e8f0;
    }
    
    /* Custom Scrollbar for Streamlit */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 242, 254, 0.2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 242, 254, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

inject_cyber_theme()

# ==========================================
# 6. IN-LINE HTML TEXT-TO-SPEECH ELEMENT
# ==========================================
def generate_speech_synthesis_element(response_text, unique_id):
    """Embed Web Speech API dynamic speech elements safely."""
    escaped_text = html.escape(response_text.replace("'", "\\'").replace("\n", " "))
    node_id = f"speech_synthesis_node_{unique_id}"
    
    # One-line layout for the button structures to prevent formatting anomalies
    element_html = f"""
    <div style="display: flex; justify-content: flex-end; margin-top: -10px; margin-bottom: 15px;">
        <button id="{node_id}" class="glow-speech-trigger" onclick="window.speechSynthesis.cancel(); var speechUtterance = new SpeechSynthesisUtterance('{escaped_text}'); speechUtterance.rate = 0.98; speechUtterance.pitch = 1.02; var systemVoices = window.speechSynthesis.getVoices(); var defaultEnglish = systemVoices.find(v => v.lang.startsWith('en')); if(defaultEnglish) speechUtterance.voice = defaultEnglish; window.speechSynthesis.speak(speechUtterance); var btnElement = document.getElementById('{node_id}'); btnElement.innerHTML = '🔊 Speaking...'; speechUtterance.onend = function() {{ btnElement.innerHTML = '🔊 Listen'; }};">🔊 Listen</button>
    </div>
    <style>
        .glow-speech-trigger {{
            background: linear-gradient(135deg, rgba(0, 242, 254, 0.05), rgba(79, 172, 254, 0.05));
            color: #00f2fe;
            border: 1px solid rgba(0, 242, 254, 0.3);
            border-radius: 10px;
            padding: 5px 14px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            font-weight: 500;
            outline: none;
        }}
        .glow-speech-trigger:hover {{
            background: linear-gradient(135deg, #00f2fe, #4facfe);
            color: #0f172a !important;
            border-color: transparent;
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.45);
            transform: translateY(-1px);
        }}
        .glow-speech-trigger:active {{
            transform: translateY(0px);
        }}
    </style>
    """
    return element_html

# ==========================================
# 7. COGNITIVE ENGINE (SIMILARITY SEARCH)
# ==========================================
def evaluate_semantic_similarity(user_query, faq_dataset, confidence_limit=0.30):
    """Normalize user input, translate to TF-IDF matrix, and evaluate matching entries."""
    # Process the dataset question fields
    faq_dataset['Normalized_Question'] = faq_dataset['Question'].apply(clean_and_tokenize)
    
    # Process user query
    normalized_query = clean_and_tokenize(user_query)
    
    # Fallback to standard lowercase split if all words were stopwords
    if not normalized_query.strip():
        normalized_query = user_query.lower().strip()
        
    try:
        # Initialize text frequency-inverse document frequency vectorizer
        tfidf_model = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        dataset_matrix = tfidf_model.fit_transform(faq_dataset['Normalized_Question'])
        
        # Transform user query to matching dimension
        query_vector = tfidf_model.transform([normalized_query])
        
        # Calculate cosine similarities
        cosine_scores = cosine_similarity(query_vector, dataset_matrix).flatten()
        
        # Retrieve highest score indices
        optimal_index = cosine_scores.argmax()
        highest_score = cosine_scores[optimal_index]
        
        # Evaluate against the parameterized boundary limit
        if highest_score >= confidence_limit:
            response_text = faq_dataset.iloc[optimal_index]['Answer']
            original_question = faq_dataset.iloc[optimal_index]['Question']
            return response_text, highest_score, original_question
        else:
            fallback_text = "I couldn't find a high-confidence match for that question. Could you please try rephrasing it or selecting one of the suggested FAQs?"
            return fallback_text, highest_score, None
            
    except Exception as exc:
        return f"⚠️ Semantic matching engine processing anomaly: {str(exc)}", 0.0, None

# ==========================================
# 8. CONVERSATIONAL SESSION CONTROLS
# ==========================================
# Track interaction sequences
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Populate fallback greeting message
if len(st.session_state.chat_history) == 0:
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "👋 Welcome! I am **AuraFAQ**, your interactive AI knowledge companion. Ask me any question about Artificial Intelligence, Machine Learning, Deep Learning, or NLP, and I will assist you! 🌌",
        "score": 1.0,
        "match": "Default greeting node",
        "id": "welcome_sequence"
    })

# Tracker for suggestions
if "chosen_suggestion" not in st.session_state:
    st.session_state.chosen_suggestion = None

def register_suggestion_selection(prompt_text):
    st.session_state.chosen_suggestion = prompt_text

# ==========================================
# 9. DYNAMIC GRAPHICAL USER INTERFACE
# ==========================================

# Side Control Dashboard
with st.sidebar:
    st.markdown("""
    <div class="avatar-wrapper">
        <div class="pulsing-avatar">🌌</div>
        <h2 style='margin: 0; font-size: 1.5rem; font-weight: 700; color: #f8fafc;'>AuraFAQ</h2>
        <p style='margin: 4px 0 0 0; color: #00f2fe; font-size: 0.72rem; letter-spacing: 1.2px; font-weight: 600; text-transform: uppercase;'>Cognitive Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section 1: System Specifications
    st.markdown("### 📊 Assistant Metrics")
    st.markdown(f"""
    <div class="system-card">
        <strong>Engine Mode:</strong> Semantic TF-IDF<br/>
        <strong>Knowledge Base:</strong> FAQ Database File<br/>
        <strong>Size:</strong> {len(faq_dataframe)} Precompiled Items<br/>
        <strong>Status:</strong> Online and Ready ✅
    </div>
    """, unsafe_allow_html=True)
    
    # Section 2: Tuning Parameters
    st.markdown("### ⚙️ Engine Parameters")
    matching_boundary = st.slider(
        "Match Confidence Cutoff 🎯", 
        min_value=0.10, 
        max_value=1.00, 
        value=0.30, 
        step=0.05,
        help="Adjust the mathematical threshold required to accept a similarity match. Higher = exact matching; Lower = broader interpretation."
    )
    
    # Section 3: Training Dataset Accordion
    st.markdown("### 📚 Knowledge Base Index")
    with st.expander("Show Available FAQ Index", expanded=False):
        st.dataframe(
            faq_dataframe[['Question']], 
            use_container_width=True, 
            hide_index=True
        )
        st.markdown(f"_Loaded: **{len(faq_dataframe)} semantic entries**_")
        
    # Section 4: Utility Controls
    st.markdown("### 🛠️ Utilities")
    sidebar_col1, sidebar_col2 = st.columns(2)
    with sidebar_col1:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chosen_suggestion = None
            st.rerun()
    with sidebar_col2:
        # Export logic implementation
        if len(st.session_state.chat_history) > 1:
            raw_chat_transcript = "\n".join([f"{entry['role'].upper()}: {entry['content']}" for entry in st.session_state.chat_history])
            st.download_button(
                label="📥 Export Chat",
                data=raw_chat_transcript,
                file_name="aurafaq_session_export.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.button("📥 Export Chat", disabled=True, use_container_width=True)

# Main Application Banner UI
st.markdown("""
<div class="dashboard-header">
    <span style="color: #00f2fe; font-weight: 600; font-size: 0.82rem; letter-spacing: 2px; text-transform: uppercase;">Cognitive NLP Matching Engine</span>
    <h1 class="main-title" style="margin-top: 5px;">AuraFAQ Knowledge Companion</h1>
    <p class="main-subtitle">An intelligent assistant built with TF-IDF Vector Spaces and Mathematical Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)

# Metrics Grid Layout
stat_col1, stat_col2, stat_col3 = st.columns(3)
with stat_col1:
    st.info(f"📂 **Active FAQ Corpus:** {len(faq_dataframe)} Items", icon="📁")
with stat_col2:
    st.success(f"⚡ **Vector Space Model:** TF-IDF Bag-of-Words", icon="🧠")
with stat_col3:
    st.warning(f"🎯 **Similarity Threshold:** {int(matching_boundary*100)}%", icon="🛡️")

st.markdown("<br/>", unsafe_allow_html=True)

# Render Chat Feed from Session History
for index, message_entry in enumerate(st.session_state.chat_history):
    chat_avatar = "🌌" if message_entry["role"] == "assistant" else "👤"
    
    with st.chat_message(message_entry["role"], avatar=chat_avatar):
        st.markdown(message_entry["content"])
        
        # Display engineering details/badging for assistant response elements
        if message_entry["role"] == "assistant" and "score" in message_entry:
            metric_score = message_entry["score"]
            matched_query_name = message_entry["match"]
            
            # Choose badge style according to validation scores
            if metric_score >= matching_boundary:
                st.markdown(
                    f"""<div class='accuracy-indicator-high'>Match Score: {metric_score:.2f} ({int(metric_score*100)}%) • Match Index: {html.escape(str(matched_query_name))}</div>""", 
                    unsafe_allow_html=True
                )
            elif metric_score > 0.0:
                st.markdown(
                    f"""<div class='accuracy-indicator-low'>Match Score: {metric_score:.2f} ({int(metric_score*100)}%) • Below Threshold ({int(matching_boundary*100)}%)</div>""", 
                    unsafe_allow_html=True
                )
            
            # Render Speech Output button element if appropriate
            if metric_score > 0.0 and message_entry["id"] != "welcome_sequence":
                st.markdown(generate_speech_synthesis_element(message_entry["content"], f"session_msg_{index}"), unsafe_allow_html=True)

# Retrieve Query from clicked suggestion chip
pending_query = st.session_state.chosen_suggestion

# Native Chat Input Component
text_input_query = st.chat_input("Enter your question here (e.g. 'What is machine learning?' or 'What is TF-IDF?')")
if text_input_query:
    pending_query = text_input_query

# Handle incoming query execution
if pending_query:
    # Clear the temporary suggestion state
    st.session_state.chosen_suggestion = None
    
    # 1. Append User Input
    with st.chat_message("user", avatar="👤"):
        st.markdown(pending_query)
        
    st.session_state.chat_history.append({
        "role": "user",
        "content": pending_query
    })
    
    # 2. Process Assistant Output
    with st.chat_message("assistant", avatar="🌌"):
        with st.spinner("🌌 Scanning Vector Spaces & Computing Similarities..."):
            matched_answer, evaluation_score, exact_match = evaluate_semantic_similarity(
                pending_query, faq_dataframe, matching_boundary
            )
            time.sleep(0.25) # Minor latency addition to emulate thought process
            
        # Stream response chunks to screen
        output_placeholder = st.empty()
        incremental_response = ""
        for word in matched_answer.split(" "):
            incremental_response += word + " "
            output_placeholder.markdown(incremental_response + "▌")
            time.sleep(0.02)
        output_placeholder.markdown(incremental_response)
        
        # Display engineering details/badge elements
        if evaluation_score >= matching_boundary:
            st.markdown(
                f"""<div class='accuracy-indicator-high'>Match Score: {evaluation_score:.2f} ({int(evaluation_score*100)}%) • Match Index: {html.escape(str(exact_match))}</div>""", 
                unsafe_allow_html=True
            )
        elif evaluation_score > 0.0:
            st.markdown(
                f"""<div class='accuracy-indicator-low'>Match Score: {evaluation_score:.2f} ({int(evaluation_score*100)}%) • Below Threshold ({int(matching_boundary*100)}%)</div>""", 
                unsafe_allow_html=True
            )
            
        # Speech synthesizer triggers
        generated_msg_id = f"session_msg_{len(st.session_state.chat_history)}"
        if evaluation_score > 0.0:
            st.markdown(generate_speech_synthesis_element(matched_answer, generated_msg_id), unsafe_allow_html=True)
            
    # Save Assistant Response to transcript logs
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": matched_answer,
        "score": float(evaluation_score),
        "match": exact_match if exact_match else "None (Low Confidence)",
        "id": generated_msg_id
    })
    st.rerun()

# Suggested chips (Quick Prompts)
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("""<div class="quick-prompts-header">💡 Frequently Asked Questions (Click to Ask)</div>""", unsafe_allow_html=True)
prompt_columns = st.columns(3)
precompiled_prompts = [
    "What is Machine Learning?",
    "What is Deep Learning?",
    "What is Cosine Similarity?",
    "What is tokenization?",
    "How can we prevent overfitting?",
    "What is the difference between stemming and lemmatization?"
]

for prompt_index, prompt_text in enumerate(precompiled_prompts):
    column_selection = prompt_index % 3
    with prompt_columns[column_selection]:
        if st.button(f"🔍 {prompt_text}", key=f"sug_btn_{prompt_index}", use_container_width=True):
            register_suggestion_selection(prompt_text)
            st.rerun()
