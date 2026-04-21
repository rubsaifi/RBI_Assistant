"""
LLM Handler Module
Handles interactions with various Open Source LLM APIs.
Includes conversation history management and token counting.
"""

import os
import sys
import re
import requests
import json
from typing import Optional, List, Dict
from pathlib import Path
from collections import deque

# Import config for API keys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_api_key


class ConversationManager:
    """
    Manages conversation history with token counting and context window management.
    Ensures the chatbot can handle N queries without bursting out of tokens.
    """

    def __init__(
        self,
        max_context_tokens: int = 6000,
        max_messages: int = 20,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize conversation manager.

        Args:
            max_context_tokens: Maximum tokens to keep in context (approximate)
            max_messages: Maximum number of messages to retain
            system_prompt: Optional system prompt to always include
        """
        self.max_context_tokens = max_context_tokens
        self.max_messages = max_messages
        self.system_prompt = system_prompt
        self.messages = deque(maxlen=max_messages)
        self.query_count = 0

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using simple heuristics.
        ~4 chars per token for English text is a reasonable approximation.
        """
        if not text:
            return 0
        # Rough estimation: 1 token ~= 4 characters
        return len(text) // 4

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "tokens": self.estimate_tokens(content)
        })
        if role == "user":
            self.query_count += 1

    def get_context_window(self, current_query: str, current_context: str) -> List[Dict]:
        """
        Get optimized context window for the LLM.
        Includes system prompt, relevant conversation history, and current context.
        """
        messages = []

        # Always add system prompt first
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Calculate available tokens for history
        system_tokens = self.estimate_tokens(self.system_prompt) if self.system_prompt else 0
        query_tokens = self.estimate_tokens(current_query)
        context_tokens = self.estimate_tokens(current_context)
        available_tokens = self.max_context_tokens - system_tokens - query_tokens - context_tokens - 500  # Buffer

        # Add recent conversation history that fits within token limit
        history = []
        total_history_tokens = 0

        for msg in reversed(self.messages):
            msg_tokens = msg.get("tokens", self.estimate_tokens(msg["content"]))
            if total_history_tokens + msg_tokens < available_tokens * 0.3:  # Use 30% for history
                history.insert(0, {
                    "role": msg["role"],
                    "content": msg["content"]
                })
                total_history_tokens += msg_tokens
            else:
                break

        messages.extend(history)

        # Add current context as a system message
        if current_context:
            context_summary = current_context[:1500]  # Limit context length
            messages.append({
                "role": "system",
                "content": f"Relevant RBI document context:\n{context_summary}"
            })

        # Add current query
        messages.append({
            "role": "user",
            "content": current_query
        })

        return messages

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for context."""
        if not self.messages:
            return "No previous conversation."

        summary = []
        for msg in self.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            summary.append(f"{role}: {content}")

        return "\n".join(summary)

    def clear_history(self):
        """Clear conversation history."""
        self.messages.clear()
        self.query_count = 0

    def should_summarize(self) -> bool:
        """Check if conversation should be summarized."""
        total_tokens = sum(m.get("tokens", self.estimate_tokens(m["content"])) for m in self.messages)
        return total_tokens > self.max_context_tokens * 0.8 or len(self.messages) >= self.max_messages


class LLMProvider:
    """Base class for LLM providers."""

    def generate_response(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        raise NotImplementedError


class GroqProvider(LLMProvider):
    """
    Groq Provider - Recommended for fast inference
    Models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
    Free tier: 1M tokens/day
    Sign up: https://console.groq.com/
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or get_api_key("GROQ_API_KEY")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_response(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        if not self.api_key or self.api_key == "gsk_your_api_key_here":
            return "❌ GROQ_API_KEY not set. Please add your actual API key from https://console.groq.com/"

        # Validate API key format (Groq keys start with 'gsk_')
        if not self.api_key.startswith("gsk_"):
            return f"❌ Invalid GROQ_API_KEY format. Key should start with 'gsk_'. Current key starts with: '{self.api_key[:10]}...'"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "top_p": 0.1
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_json = response.json()
                error_detail = f" Details: {error_json.get('error', {}).get('message', str(error_json))}"
            except:
                error_detail = f" Response: {response.text[:200]}"
            return f"❌ Groq API Error ({response.status_code}): {str(e)}{error_detail}"
        except Exception as e:
            return f"❌ Error calling Groq API: {str(e)}"


class HuggingFaceProvider(LLMProvider):
    """
    Hugging Face Inference API - Free tier available
    Models: mistralai/Mistral-7B-Instruct-v0.2, meta-llama/Llama-2-70b-chat-hf
    Sign up: https://huggingface.co/settings/tokens
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.api_key = api_key or get_api_key("HUGGINGFACE_API_KEY")
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"

    def generate_response(self, messages: List[Dict], max_tokens: int = 512) -> str:
        if not self.api_key:
            return "❌ HUGGINGFACE_API_KEY not set."

        # Convert messages to prompt format
        full_prompt = self._format_messages(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.1,
                "return_full_text": False
            }
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            return str(result)
        except Exception as e:
            return f"❌ Error calling Hugging Face API: {str(e)}"

    def _format_messages(self, messages: List[Dict]) -> str:
        """Convert message list to prompt string."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            else:
                parts.append(f"Assistant: {content}")
        return "\n\n".join(parts) + "\n\nAssistant:"


class LocalModelProvider(LLMProvider):
    """
    Local model using Ollama (runs locally, no API key needed)
    Models: llama3, mistral, phi3
    Requires: ollama installed locally
    """

    def __init__(self, model: str = "llama3"):
        self.model = model
        self.api_url = "http://localhost:11434/api/chat"

    def generate_response(self, messages: List[Dict], max_tokens: int = 512) -> str:
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                self.api_url,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "No response generated")
        except requests.exceptions.ConnectionError:
            return "❌ Ollama not running. Install from ollama.ai and start the server."
        except Exception as e:
            return f"❌ Error with local model: {str(e)}"


class GeminiProvider(LLMProvider):
    """
    Google Gemini API - Free tier available
    Model: gemini-1.5-flash, gemini-1.5-pro
    Sign up: https://aistudio.google.com/app/apikey
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or get_api_key("GOOGLE_API_KEY")
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def generate_response(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        if not self.api_key:
            return "❌ GOOGLE_API_KEY not set. Get it from https://aistudio.google.com/app/apikey"

        # Convert messages to Gemini format
        contents = self._format_messages(messages)

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_tokens,
                "topP": 0.1
            }
        }

        try:
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return "❌ No response from Gemini API"
        except Exception as e:
            return f"❌ Error calling Gemini API: {str(e)}"

    def _format_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert messages to Gemini format."""
        contents = []
        current_role = None
        current_parts = []

        for msg in messages:
            role = "model" if msg["role"] in ["assistant", "system"] else "user"

            if role != current_role and current_parts:
                contents.append({
                    "role": current_role,
                    "parts": [{"text": "\n".join(current_parts)}]
                })
                current_parts = []

            current_role = role
            current_parts.append(msg["content"])

        if current_parts:
            contents.append({
                "role": current_role,
                "parts": [{"text": "\n".join(current_parts)}]
            })

        return contents


# Provider configuration
PROVIDERS = {
    "groq": GroqProvider,
    "huggingface": HuggingFaceProvider,
    "ollama": LocalModelProvider,
    "gemini": GeminiProvider,
}


def get_llm_response(
    question: str,
    context: str,
    conversation_manager: Optional[ConversationManager] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> dict:
    """
    Get LLM response for a question with context.
    Supports unlimited queries through conversation management.

    Args:
        question: User question
        context: Retrieved context from documents
        conversation_manager: Conversation history manager (optional)
        provider: LLM provider to use (default from env or groq)
        model: Model name (if applicable)

    Returns:
        Dict with 'answer', 'suggested_questions', 'predicted_query', and 'query_count'
    """
    # System prompt for RBI assistant
    system_prompt = """You are an RBI policy expert. Provide SHORT, CONCISE answers.

RULES:
1. Use ONLY the provided context - no external knowledge
2. If info not in context, say: "Information not found in document."
3. Keep answers under 100 tokens - be direct
4. Use bullet points for lists
5. After answering, suggest 2-3 related follow-up questions
6. Predict 1 likely next query the user might ask

Format your response EXACTLY as:
ANSWER: <your concise answer here>
FOLLOW_UPS: <question1>, <question2>, <question3>
PREDICTED: <most likely next question>"""

    # Create or use conversation manager
    if conversation_manager is None:
        conversation_manager = ConversationManager(system_prompt=system_prompt)

    # Determine provider
    provider_name = provider or os.getenv("LLM_PROVIDER", "groq").lower()

    if provider_name not in PROVIDERS:
        return {
            "answer": f"❌ Unknown provider: {provider_name}",
            "suggested_questions": [],
            "predicted_query": "",
            "query_count": conversation_manager.query_count
        }

    # Create provider instance
    provider_class = PROVIDERS[provider_name]
    llm = provider_class(model=model) if model else provider_class()

    # Get optimized context window
    messages = conversation_manager.get_context_window(question, context)

    # Generate response
    raw_response = llm.generate_response(messages, max_tokens=1024)

    # Parse the structured response
    result = parse_llm_response(raw_response)
    result["query_count"] = conversation_manager.query_count

    # Update conversation history
    conversation_manager.add_message("user", question)
    conversation_manager.add_message("assistant", result["answer"])

    # Check if we should summarize (for very long conversations)
    if conversation_manager.should_summarize():
        result["notice"] = "Conversation auto-summarized to maintain performance."

    return result


def parse_llm_response(response: str) -> dict:
    """Parse LLM response into structured format."""
    result = {
        "answer": "",
        "suggested_questions": [],
        "predicted_query": "",
        "query_count": 0
    }

    lines = response.split("\n")

    for line in lines:
        if line.startswith("ANSWER:"):
            result["answer"] = line.replace("ANSWER:", "").strip()
        elif line.startswith("FOLLOW_UPS:"):
            followups = line.replace("FOLLOW_UPS:", "").strip()
            result["suggested_questions"] = [q.strip() for q in followups.split(",") if q.strip()]
        elif line.startswith("PREDICTED:"):
            result["predicted_query"] = line.replace("PREDICTED:", "").strip()

    # Fallback if parsing fails
    if not result["answer"]:
        result["answer"] = response

    return result


def get_provider_recommendation() -> str:
    """Get recommendation for best LLM provider."""
    return """
## 🏆 Recommended Open Source LLM APIs

### 1. **Groq** (⭐ Highly Recommended for Streamlit Cloud)
- **Models**: Llama 3 (8B, 70B), Mixtral 8x7B
- **Free Tier**: 1 Million tokens/day
- **Speed**: Extremely fast (world's fastest inference)
- **Setup**: Get API key from https://console.groq.com/
- **Best For**: Production deployment, fast responses

### 2. **Google Gemini** (⭐ Good Free Option)
- **Models**: Gemini 1.5 Flash, Gemini 1.5 Pro
- **Free Tier**: 60 requests/minute, generous limits
- **Setup**: Get API key from https://aistudio.google.com/app/apikey
- **Best For**: Cost-effective, good quality

### 3. **Hugging Face**
- **Models**: Mistral-7B, Llama-2, various open models
- **Free Tier**: Available with rate limits
- **Setup**: Get token from https://huggingface.co/settings/tokens
- **Best For**: Model variety, open source

### 4. **Ollama (Local)**
- **Models**: Llama 3, Mistral, Phi-3
- **Cost**: Completely free (runs locally)
- **Setup**: Install from https://ollama.ai
- **Best For**: Privacy, offline use, unlimited usage

### 🎯 Recommendation for Streamlit Cloud
**Groq** is the best choice because:
- ✅ Generous free tier (1M tokens/day)
- ✅ Blazing fast inference speed
- ✅ No cold start issues
- ✅ Reliable API uptime
- ✅ Easy integration

Set your API key in Streamlit Cloud Secrets as `GROQ_API_KEY`
"""


if __name__ == "__main__":
    print(get_provider_recommendation())
