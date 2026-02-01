import numpy as np
import faiss
import hashlib
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
CHAT_MODEL = "llama-3.3-70b-versatile"


def mock_embed_text(text: str) -> list:
    """Create mock embedding by hashing text (for quota-limited testing)"""
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert to 768-dimensional vector
    embedding = []
    for i in range(768):
        byte_val = hash_bytes[i % len(hash_bytes)]
        embedding.append((byte_val - 128) / 128.0)
    
    return embedding


def embed_query(query: str):
    """Generate embedding for user query - uses mock to avoid API quotas"""
    embedding = mock_embed_text(query)
    vec = np.array([embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def retrieve(query, index, chunks, k=4):
    """Retrieve top-k most relevant chunks for query"""
    qvec = embed_query(query)
    _, ids = index.search(qvec, k)

    results = []
    for i in ids[0]:
        if i != -1:
            results.append(chunks[i])

    return results


def generate_answer(user_question, retrieved_chunks):
    """Generate answer using Grok with retrieved context"""
    context = "\n\n".join(retrieved_chunks)

    try:
        prompt = f"""You are an Insurance Agency Customer Care assistant. 
Use only the provided context to answer the question. 
If the answer is not present in the context, say you do not have that information and offer to connect them with human support.

Context:
{context}

Question:
{user_question}"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful Insurance Agency Customer Care assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": CHAT_MODEL,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        return f"Error calling Groq API: {str(e)}"
    except Exception as e:
        return f"Error generating answer: {str(e)}"
