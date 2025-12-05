# models/groq_llama.py

from groq import Groq
from dotenv import load_dotenv
import os

class GroqLlama:
    """
    Wrapper for Groq-hosted Llama models.
    """

    def __init__(self, model_name="llama-3.1-8b-instant"):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("GROQ_API_KEY is missing from .env")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def ask(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content
