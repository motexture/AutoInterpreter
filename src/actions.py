import ssl
import os
import yaml

from openai import OpenAI

ssl._create_default_https_context = ssl._create_unverified_context

class Actions:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), '../config/config.yaml'), 'r') as file:
            self.config = yaml.safe_load(file)

        self.client = OpenAI(base_url=self.config['api']['base_url'], api_key=self.config['api']['api_key'])
        self.model = self.config['model']['model']
        self.temperature = self.config['model']['temperature']

    def inference(self, text: str) -> str:
        system = {
            "role": "system",
            "content": (
                "You are a smart and efficient AI agent. You will always answer only with the content required and never include any additional comments or explanations."
            )
        }
        
        prompt = [system]
        prompt.append(
            {"content": f"{text}", "role": "user"},
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=self.temperature,
            stream=True,
        )

        response = ""
        try:
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    response += chunk.choices[0].delta.content
        except:
            pass
        
        return response.strip()
