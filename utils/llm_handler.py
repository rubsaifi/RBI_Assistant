"""
LLM Handler Module
Handles interactions with various Open Source LLM APIs.
"""

import os
import requests
from typing import Optional, List, Dict
from collections import deque
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()


# ==============================
# Conversation Manager
# ==============================
class ConversationManager:
    def __init__(
        self,
        max_context_tokens: int = 6000,
        max_messages: int = 20,
        system_prompt: Optional[str] = None
    ):
        self.max_context_tokens = max_context_tokens
        self.max_messages = max_messages
        self.system_prompt = system_prompt
        self.messages = deque(maxlen=max_messages)
        self.query_count = 0

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4 if text else 0

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "tokens": self.estimate_tokens(content)
        })
        if role == "user":
            self.query_count += 1

    def get_context_window(self, current_query: str, current_context: str) -> List[Dict]:
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        system_tokens = self.estimate_tokens(self.system_prompt or "")
        query_tokens = self.estimate_tokens(current_query)
        context_tokens = self.estimate_tokens(current_context)

        available_tokens = self.max_context_tokens - system_tokens - query_tokens - context_tokens - 500

        history = []
        total_tokens = 0

        for msg in reversed(self.messages):
            msg_tokens = msg["tokens"]
            if total_tokens + msg_tokens < available_tokens * 0.3:
                history.insert(0, {"role": msg["role"], "content": msg["content"]})
                total_tokens += msg_tokens
            else:
                break

        messages.extend(history)

        if current_context:
            messages.append({
                "role": "system",
                "content": f"Relevant Context:\n{current_context[:1500]}"
            })

        messages.append({"role": "user", "content": current_query})

        return messages

    def should_summarize(self) -> bool:
        total_tokens = sum(m["tokens"] for m in self.messages)
        return total_tokens > self.max_context_tokens * 0.8


# ==============================
# Providers
# ==============================
class LLMProvider:
    def generate_response(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        raise NotImplementedError


class GroqProvider(LLMProvider):
    def __init__(self, model="llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_response(self, messages, max_tokens=1024):
        if not self.api_key:
            return "❌ GROQ_API_KEY missing in .env"

        try:
            res = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Groq Error: {str(e)}"


class HuggingFaceProvider(LLMProvider):
    def __init__(self, model="mistralai/Mistral-7B-Instruct-v0.2"):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.url = f"https://api-inference.huggingface.co/models/{model}"

    def generate_response(self, messages, max_tokens=512):
        if not self.api_key:
            return "❌ HUGGINGFACE_API_KEY missing"

        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        try:
            res = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": prompt}
            )
            return res.json()[0]["generated_text"]
        except Exception as e:
            return f"❌ HF Error: {str(e)}"


class GeminiProvider(LLMProvider):
    def __init__(self, model="gemini-1.5-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def generate_response(self, messages, max_tokens=1024):
        if not self.api_key:
            return "❌ GOOGLE_API_KEY missing"

        contents = [{"role": "user", "parts": [{"text": m["content"]}]} for m in messages]

        try:
            res = requests.post(
                f"{self.url}?key={self.api_key}",
                json={"contents": contents}
            )
            return res.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"❌ Gemini Error: {str(e)}"


class LocalModelProvider(LLMProvider):
    def __init__(self, model="llama3"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

    def generate_response(self, messages, max_tokens=512):
        try:
            res = requests.post(self.url, json={
                "model": self.model,
                "messages": messages
            })
            return res.json()["message"]["content"]
        except Exception:
            return "❌ Ollama not running"


PROVIDERS = {
    "groq": GroqProvider,
    "huggingface": HuggingFaceProvider,
    "gemini": GeminiProvider,
    "ollama": LocalModelProvider,
}


# ==============================
# Main Function
# ==============================
def get_llm_response(
    question: str,
    context: str,
    conversation_manager: Optional[ConversationManager] = None,
    provider: str = "groq"
) -> dict:

    system_prompt = """You are an RBI policy expert.
Answer ONLY from given context.
Keep answers short and precise."""

    if conversation_manager is None:
        conversation_manager = ConversationManager(system_prompt=system_prompt)

    llm = PROVIDERS[provider]()

    messages = conversation_manager.get_context_window(question, context)

    response = llm.generate_response(messages)

    # Save history
    conversation_manager.add_message("user", question)
    conversation_manager.add_message("assistant", response)

    return {
        "answer": response,
        "query_count": conversation_manager.query_count
    }


# ==============================
# Test Run
# ==============================
if __name__ == "__main__":
    cm = ConversationManager()

    result = get_llm_response(
        "What is repo rate?",
        "Repo rate is 6.5%",
        conversation_manager=cm
    )

    print(result)