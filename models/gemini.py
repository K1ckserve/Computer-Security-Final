# models/gemini.py

import time
import google.generativeai as genai
from dotenv import load_dotenv
import os

class GeminiFlash:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("GEMINI_API_KEY is missing from .env")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def ask(self, prompt: str) -> str:
        # simple throttle: 4 seconds between calls ≈ 15/min
        time.sleep(4)
        response = self.model.generate_content(prompt)
        return response.text
