# 🏛️ RBI Master Policy Assistant

An AI-powered chatbot built with Python and Streamlit that helps users resolve queries related to RBI (Reserve Bank of India) Master Circular policies.

## 📋 Features

- 🤖 **AI-Powered Q&A**: Ask questions about RBI policies and get accurate answers
- 📚 **RAG-Based**: Uses Retrieval-Augmented Generation for precise responses
- 💬 **Unlimited Queries**: Smart conversation management prevents token limits
- 🎨 **Themed UI**: Professional RBI-themed interface with custom colors
- ⚡ **Fast Responses**: Optimized for quick inference using Groq API
- 🔒 **Privacy Focused**: No data stored, all queries processed in real-time

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd Prod_RBI_Assistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

**For Local Development:**
Edit `.streamlit/secrets.toml` with your API keys:
```toml
GROQ_API_KEY = "your-api-key-here"
LLM_PROVIDER = "groq"
```

**For Production (Streamlit Cloud):**
Add secrets in Streamlit Dashboard (see Deployment section)

**Supported Providers:**

| Provider | Free Tier | Speed | Get API Key |
|----------|-----------|-------|-------------|
| **Groq** ⭐ | 1M tokens/day | ⚡⚡⚡ | https://console.groq.com/ |
| **Gemini** | 60 req/min | ⚡⚡ | https://aistudio.google.com/app/apikey |
| **HuggingFace** | Rate limited | ⚡ | https://huggingface.co/settings/tokens |
| **Ollama** | Unlimited | Local | https://ollama.ai (no key needed) |

### 3. Run the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## 📁 Project Structure

```
Prod_RBI_Assistant/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── utils/
│   ├── __init__.py
│   ├── pdf_loader.py     # PDF processing module
│   ├── rag_engine.py     # RAG (Retrieval) engine
│   └── llm_handler.py    # LLM API handler
└── Docs/
    └── rbi_master.pdf    # RBI Master Circular document
```

## 🌐 Deployment on Streamlit Cloud

### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files EXCEPT secrets
git add .

# Commit
git commit -m "Initial commit - RBI Assistant with voice input"

# Create main branch and push
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

> ⚠️ **Important**: The `.streamlit/secrets.toml` file is in `.gitignore` and will NOT be pushed to GitHub. This keeps your API keys secure!

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set main file path: `app.py`
6. **Add Secrets** (Click "Advanced settings" → "Secrets"):

   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   LLM_PROVIDER = "groq"
   ```

7. Click **"Deploy!"**

### Step 3: Share Your App
- App URL: `https://your-app-name.streamlit.app`
- The app will be publicly accessible (or private based on your settings)

## 🎨 Theme Customization

The chatbot uses a custom RBI-themed color scheme:

- **User Messages**: Purple gradient (`#667eea` → `#764ba2`)
- **Bot Messages**: Pink/Orange gradient (`#f093fb` → `#f5576c`)
- **Header**: Navy blue (`#1e3a5f` → `#2c5282`)
- **Accent**: Golden (`#d69e2e`)

To modify colors, edit the CSS in `app.py` in the `load_custom_css()` function.

## 🔧 Supported LLM Providers

| Provider | Free Tier | Speed | Best For |
|----------|-----------|-------|----------|
| **Groq** | 1M tokens/day | ⚡⚡⚡ Fastest | Production |
| **Gemini** | 60 req/min | ⚡⚡ Fast | Cost-effective |
| **HuggingFace** | Rate limited | ⚡ Moderate | Model variety |
| **Ollama** | Unlimited | Depends on hardware | Privacy |

## 📝 Sample Questions

- "What are the KYC guidelines for banks?"
- "Explain the Basel III capital adequacy requirements"
- "What is the priority sector lending target?"
- "Tell me about the RBI guidelines on lending rates"
- "What are the requirements for opening a bank account?"

## 🛠️ Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black utils/ app.py
flake8 utils/ app.py
```

## 📄 License

This project is for educational purposes. RBI Master Circular content is © Reserve Bank of India.

## 🤝 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Open an issue on GitHub
3. Contact the development team

## 🔍 Troubleshooting

**Issue**: "GROQ_API_KEY not set"
- **Solution**: Set the environment variable or add to Streamlit Cloud secrets

**Issue**: "PDF not found"
- **Solution**: Ensure `rbi_master.pdf` is in the `Docs/` folder

**Issue**: "Module not found"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: "Model loading slowly"
- **Solution**: First run downloads embedding model (~80MB), subsequent runs are faster

---

Built with ❤️ using Streamlit and Open Source AI
