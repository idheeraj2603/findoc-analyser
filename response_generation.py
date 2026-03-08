from groq import Groq
from constants import GROQ_API_KEY, GROQ_MODEL
client = Groq(api_key=GROQ_API_KEY)


def generate_response(query: str, context: str) -> str:
    """
    Generate an answer to the query using the retrieved context chunks.
    Falls back to an error message if the LLM call fails.
    """
    prompt = f"""
You are a financial analyst assistant helping users understand 10-K SEC filings.

Use the context below to answer the question. If the answer is not in the context, say "I couldn't find that information in the document."

Context:
{context}

Question: {query}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a financial analyst assistant. Answer only from the provided context. If not found, say 'I couldn't find that information in the document.'"},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Response generation failed: {e}")
        return "Sorry, I was unable to generate a response. Please check your Groq API key."
