# rag_engine.py

from openai import OpenAI
from utils import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_rag_response(query, context_docs):
    # Join your retrieved docs into one context string
    context = "\n\n".join(context_docs)

    prompt = f"""
You are SkySense, an expert assistant for Indian and global weather. Use the following context to answer the user query as accurately as possible.

Context:
{context}

Query: {query}

Respond using the context. Cite facts if possible. Do not hallucinate or make up details beyond the content.
"""

    print(prompt)

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content