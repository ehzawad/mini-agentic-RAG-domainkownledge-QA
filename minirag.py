import os
import requests
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")
EMBED_DEPLOYMENT = os.getenv("EMBED_DEPLOYMENT")

CHAT_API_VERSION = os.getenv("CHAT_API_VERSION")
EMBED_API_VERSION = os.getenv("EMBED_API_VERSION")

CHAT_URL = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{CHAT_DEPLOYMENT}/chat/completions?api-version={CHAT_API_VERSION}"
EMBED_URL = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{EMBED_DEPLOYMENT}/embeddings?api-version={EMBED_API_VERSION}"

HEADERS = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_KEY,
}

def azure_chat(prompt: str, system_prompt: str = "You are a helpful mini-agentic RAG system designed to provide domain-specific (NIST security) Q&A responses. You leverage knowledge-based embeddings to deliver accurate, tailored answers using the GPT-4o model.") -> str:
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    }
    resp = requests.post(CHAT_URL, headers=HEADERS, json=body)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def azure_chat_with_tools(
    messages: list[dict],
    tools: list[dict] = None,
    system_prompt: str = "You are a helpful retrieval agent who uses available tools to answer user queries. You self-reflect when needed to ensure accuracy, then provide a polished, comprehensive response."
) -> dict:
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
    body = {"messages": full_messages}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    
    resp = requests.post(CHAT_URL, headers=HEADERS, json=body)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]

def azure_embed(text: str) -> list[float]:
    body = {
        "input": text
    }
    resp = requests.post(EMBED_URL, headers=HEADERS, json=body)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]

if __name__ == "__main__":
    # reply = azure_chat("give me a short plan on building a mini agentic RAG systems for domain Knowledge QA")
    reply = azure_chat("Hi")
    print("Chat reply:", reply)

    sentence = "Azure OpenAI provides powerful language models."
    vec = azure_embed(sentence)
    print("Embedding length:", len(vec))
    print("First 5 dims:", vec[:5])
    # print("The full vector dims is stored", vec)