import ollama

class OllamaDeepseek:
    def __init__(self, model_name="deepseek-r1:8b"):
        self.model_name = model_name

    def ask(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
