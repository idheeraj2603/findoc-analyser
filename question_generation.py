import json
from groq import Groq
from constants import GROQ_API_KEY, GROQ_MODEL
client = Groq(api_key=GROQ_API_KEY)


def generate_subquestions(query: str) -> list[str]:
    """
    Break a broad query into sub-questions to improve retrieval.
    Falls back to the original query if anything goes wrong.
    """
    prompt = f"""
You are a financial analyst assistant.

A user asked: "{query}"

Break this into 3 specific sub-questions to help retrieve relevant information from a 10-K financial document.

Return ONLY a JSON object like this: {{"questions": ["q1", "q2", "q3"]}}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        questions = json.loads(raw).get("questions", [])
        
        # always include original query for full coverage
        if query not in questions:
            questions.insert(0, query)

        return questions

    except Exception as e:
        print(f"Sub-question generation failed: {e}")
        return [query]
