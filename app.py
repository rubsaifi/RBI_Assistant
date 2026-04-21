"""
RBI Master Policy AI Assistant
A Streamlit-based chatbot for querying RBI Master Circular documents.
Supports unlimited queries with conversation history management and voice input.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import random

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.pdf_loader import load_and_process_pdf
from utils.rag_engine import RAGEngine
from utils.llm_handler import get_llm_response, ConversationManager

# Page configuration
st.set_page_config(
    page_title="RBI Master Policy Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dynamic color generator for messages
def get_dynamic_color(seed=None):
    """Generate a random pleasing color for message backgrounds."""
    if seed is not None:
        random.seed(seed)

    # Predefined pleasing gradients for user messages (cool colors)
    user_gradients = [
        "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
        "linear-gradient(135deg, #5B86E5 0%, #36D1DC 100%)",
        "linear-gradient(135deg, #2E3192 0%, #1BFFFF 100%)",
        "linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%)",
    ]

    # Predefined pleasing gradients for bot messages (warm colors)
    bot_gradients = [
        "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
        "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "linear-gradient(135deg, #f6d365 0%, #fda085 100%)",
        "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)",
        "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
        "linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%)",
        "linear-gradient(135deg, #FEE140 0%, #FA709A 100%)",
    ]

    return {
        "user": random.choice(user_gradients),
        "bot": random.choice(bot_gradients)
    }

# Custom CSS with white background and dynamic colors
def load_custom_css():
    st.markdown("""
    <style>
    /* Black background */
    .stApp {
        background-color: #0d1117 !important;
    }

    .main {
        background-color: #0d1117 !important;
    }

    .main .block-container {
        background-color: #0d1117 !important;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2c5282 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #d69e2e !important;
        font-size: 1.2rem;
        font-weight: 500;
    }

    /* Chat container - dark background */
    .chat-container {
        background-color: #161b22;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Message container for proper spacing */
    .message-container {
        margin-bottom: 2rem;
    }

    /* Message labels */
    .message-label {
        font-size: 0.75rem;
        color: #718096;
        margin-bottom: 0.25rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .user-label {
        text-align: right;
        margin-right: 1rem;
    }

    .bot-label {
        text-align: left;
        margin-left: 1rem;
        color: #d69e2e;
    }

    /* Input area styling */
    .stTextInput > div > div > input {
        background-color: #f7fafc;
        border: 2px solid #e2e8f0;
        border-radius: 25px;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        color: #2d3748;
    }

    .stTextInput > div > div > input:focus {
        border-color: #2c5282;
        box-shadow: 0 0 0 3px rgba(44, 82, 130, 0.1);
    }

    /* Form button */
    .stFormSubmitButton > button {
        background: linear-gradient(90deg, #1e3a5f 0%, #2c5282 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.4);
    }

    /* Voice button container */
    .voice-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }

    /* Info cards */
    .info-card {
        background-color: white;
        border-left: 4px solid #d69e2e;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #8b949e;
        font-size: 0.9rem;
        border-top: 1px solid #30363d;
        margin-top: 2rem;
        background-color: #161b22;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}

    /* Sidebar styling - Black background with white text */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
    }

    [data-testid="stSidebar"] .css-1d391kg {
        background-color: #0d1117 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #0d1117 !important;
    }

    /* Sidebar text colors */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] p {
        color: #c9d1d9 !important;
    }

    [data-testid="stSidebar"] span {
        color: #c9d1d9 !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9 !important;
    }

    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-color: #30363d !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(90deg, #238636 0%, #2ea043 100%) !important;
        color: white !important;
        border: none !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(90deg, #2ea043 0%, #3fb950 100%) !important;
    }

    /* Custom scrollbar - dark theme */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0d1117;
    }

    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "message_colors" not in st.session_state:
        st.session_state.message_colors = {}

    # Initialize conversation manager for unlimited queries
    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager(
            max_context_tokens=6000,
            max_messages=20,
            system_prompt="""You are an RBI policy expert. Provide SHORT, CONCISE answers.

RULES:
1. Use ONLY the provided context - no external knowledge
2. If info not in context, say: "Information not found in document."
3. Keep answers under 100 tokens - be direct
4. Use bullet points for lists

Give a natural, conversational response without any prefixes or labels."""
        )

    if "rag_engine" not in st.session_state:
        with st.spinner("🔄 Loading RBI Master Document..."):
            try:
                # Load PDF from Docs folder
                pdf_path = Path(__file__).parent / "Docs" / "rbi_master.pdf"

                if not pdf_path.exists():
                    st.error(f"❌ PDF not found at {pdf_path}")
                    st.session_state.rag_engine = None
                    return

                # Process PDF and initialize RAG
                documents = load_and_process_pdf(str(pdf_path))
                st.session_state.rag_engine = RAGEngine(documents)
                st.session_state.documents_loaded = True

            except Exception as e:
                st.error(f"❌ Error loading documents: {str(e)}")
                st.session_state.rag_engine = None


def get_message_style(role, message_id):
    """Get dynamic style for a message."""
    if message_id not in st.session_state.message_colors:
        colors = get_dynamic_color(seed=message_id)
        st.session_state.message_colors[message_id] = colors

    colors = st.session_state.message_colors[message_id]
    bg = colors["user"] if role == "user" else colors["bot"]

    return f"""
        background: {bg};
        color: black;
        padding: 1rem 1.5rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        {'margin-left: auto;' if role == "user" else 'margin-right: auto;'}
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-size: 1rem;
        line-height: 1.6;
    """


def display_chat_history():
    """Display chat history with suggested questions and query prediction."""
    for idx, message in enumerate(st.session_state.messages):
        message_id = f"{message['role']}_{idx}"
        style = get_message_style(message["role"], message_id)

        if message["role"] == "user":
            st.markdown(f"""
                <div class="message-container">
                    <div class="message-label user-label">You</div>
                    <div style="{style}">{message['content']}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Display answer
            st.markdown(f"""
                <div class="message-container">
                    <div class="message-label bot-label">🏛️ RBI Assistant</div>
                    <div style="{style}">{message['content']}</div>
                </div>
            """, unsafe_allow_html=True)

            # Display suggested questions
            suggested = message.get("suggested_questions", [])
            if suggested:
                st.markdown("""
                <div style="
                    background-color: #161b22;
                    border-left: 3px solid #d69e2e;
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                ">
                    <strong style="color: #d69e2e;">💡 Suggested Questions:</strong>
                </div>
                """, unsafe_allow_html=True)

                for q in suggested:
                    if st.button(f"▸ {q}", key=f"suggested_{idx}_{q[:20]}"):
                        process_user_question(q)
                        st.rerun()

            # Display predicted query
            predicted = message.get("predicted_query", "")
            if predicted:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #1f6feb20 0%, #23863620 100%);
                    border: 2px solid #238636;
                    padding: 1rem;
                    border-radius: 10px;
                    margin: 1rem 0;
                ">
                    <strong style="color: #3fb950;">🔮 Predicted Next Question:</strong>
                    <div style="
                        background-color: #0d1117;
                        padding: 0.75rem 1rem;
                        border-radius: 8px;
                        margin-top: 0.5rem;
                        color: #7ee787;
                    ">
                        {predicted}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Ask: {predicted}", key=f"predicted_{idx}"):
                    process_user_question(predicted)
                    st.rerun()


# Voice input component removed as per user request


def process_user_question(question: str):
    """Process user question and generate response."""
    if not question or not question.strip():
        return

    question = question.strip()

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Generate response with conversation management
    with st.spinner("🏛️ Analyzing RBI Master Circular..."):
        try:
            if st.session_state.rag_engine:
                context = st.session_state.rag_engine.get_relevant_context(question)

                # Use conversation manager for unlimited queries
                response_data = get_llm_response(
                    question=question,
                    context=context,
                    conversation_manager=st.session_state.conversation_manager
                )
            else:
                response_data = {
                    "answer": "❌ Document not loaded. Please check the PDF file.",
                    "suggested_questions": [],
                    "predicted_query": "",
                    "query_count": 0
                }

            # Add assistant message with structured response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data["answer"],
                "suggested_questions": response_data.get("suggested_questions", []),
                "predicted_query": response_data.get("predicted_query", ""),
                "query_count": response_data.get("query_count", 0)
            })

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "suggested_questions": [],
                "predicted_query": "",
                "query_count": 0
            })


def handle_user_input():
    """Process user input with Enter key submission."""
    with st.container():
        st.markdown("<br>", unsafe_allow_html=True)

        # Use st.form for Enter key submission
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Ask your question about RBI Master Policy...",
                key="input_question",
                placeholder="e.g., What are the KYC guidelines for banks?",
                label_visibility="collapsed"
            )

            # Submit button
            submitted = st.form_submit_button("🔍 Get Answer", use_container_width=True)

        # Process on form submission (Enter key or button click)
        if submitted:
            if user_question and user_question.strip():
                process_user_question(user_question.strip())
                st.rerun()


def sidebar_content():
    """Display sidebar content."""
    with st.sidebar:
        st.title("🏛️ RBI Master Policy Assistant")
        st.markdown("---")

        st.markdown("""
        ### 📋 About
        This AI assistant helps you query the **RBI Master Circular** documents
        and get accurate answers based on official RBI policies.

        ### 🎯 Features
        - 🔍 **Smart Search**: Find relevant policy information
        - 💬 **Unlimited Queries**: Smart conversation history management
        - 🤖 **AI-Powered**: Uses advanced language models
        - 📚 **Document-Based**: Answers from official RBI documents
        - ⚡ **Real-Time**: Instant responses to your queries

        ### 📝 Sample Questions
        - What are the KYC norms for banks?
        - Explain the RBI guidelines on lending rates
        - What is the priority sector lending requirement?
        - Tell me about the Basel III capital adequacy norms
        """)

        st.markdown("---")

        # Show query count if available
        if "messages" in st.session_state and st.session_state.messages:
            query_count = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.markdown(f"<p style='color: #3fb950;'>✅ Queries this session: {query_count}</p>", unsafe_allow_html=True)

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.message_colors = {}
            if "conversation_manager" in st.session_state:
                st.session_state.conversation_manager.clear_history()
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #8b949e; font-size: 0.8rem;">
            <p style="color: #8b949e;">Powered by Open Source LLM</p>
            <p style="color: #8b949e;">Data: RBI Master Circular</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Load custom CSS
    load_custom_css()

    # Initialize session state
    initialize_session_state()

    # Sidebar
    sidebar_content()

    # Main content area
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ RBI Master Policy Assistant</h1>
        <p>Your AI-powered guide to Reserve Bank of India Master Circulars</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if documents loaded successfully
    if st.session_state.get("documents_loaded"):
        st.success("✅ RBI Master Document loaded successfully!", icon="📚")
    elif st.session_state.get("rag_engine") is None:
        st.warning("⚠️ Could not load RBI Master Document. Some features may be limited.", icon="⚠️")

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display welcome message if no chat history
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #c9d1d9; background-color: #161b22; border-radius: 15px;">
            <h3 style="color: #58a6ff;">👋 Welcome to RBI Master Policy Assistant!</h3>
            <p style="font-size: 1.1rem; margin-top: 1rem; color: #8b949e;">
                Ask me anything about RBI Master Circulars, policies, and guidelines.<br>
                I'm here to help you understand banking regulations and compliance requirements.
            </p>
            <div style="margin-top: 2rem;">
                <span style="background: #238636; color: white; padding: 0.5rem 1rem; border-radius: 20px; margin: 0.5rem; display: inline-block;">
                    💬 Type your question
                </span>
                <span style="background: #8957e5; color: white; padding: 0.5rem 1rem; border-radius: 20px; margin: 0.5rem; display: inline-block;">
                    🔍 Get instant answers
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display chat history
    display_chat_history()

    st.markdown('</div>', unsafe_allow_html=True)

    # Handle user input with Enter key
    handle_user_input()


if __name__ == "__main__":
    main()
