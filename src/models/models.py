import torch
import ssl

from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

TOKEN_TO_WORD_RATIO = 1.33
SAFETY_MARGIN = 1.2

class Models:
    def __init__(self, config):
        self.client = OpenAI(base_url=config['api']['base_url'], api_key=config['api']['api_key'])
        self.model = config['model']['model']
        self.device = config['model']['device']
        self.temperature = config['model']['temperature']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['classifier'])
        self.classifier = AutoModelForSequenceClassification.from_pretrained(config['model']['classifier']).to(self.device)
        self.classifier_max_seq_length = config['model']['classifier_context_length']

    def get_task_builder_system_prompt(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                "You are an extremely smart AI agent known as the Task Builder. Your role is to explain to the Task Instructor agent what the user wants."
                "\n- Ensure the explanation is concise and easy to understand."
                "\n- Avoid at all costs giving instructions on how to solve it."
                "\n- Start with 'The user wants' words."
                "\n- Do not exceed one-two paragraphs."
                "\n- Include links if relevant."
            )
        }

    def get_task_instructor_system_prompt(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                "You are an extremely smart AI agent known as the Task Instructor. Your role is to analyze the previous content and provide exhaustive step-by-step instructions to help a Task Coder agent create Python code that will be executed immediately to complete the task itself."
                "\n\nImportant points:"
                "\n- If any, learn from past instructions and mistakes to improve your current instructions."
                "\n- Never include libraries that require an API key unless the task explicitly requires it."
                "\n- Never instruct to write code that may require human intervention such as requiring inputs or completing pieces of code."
                "\n- Provide the Task Coder agent detailed instructions for installing necessary packages with commands like `os.system('pip3 install requests --quiet')` before the imports."
                "\n- Instruct the Task Coder agent to write detailed and comprehensive code."
                "\n- Instruct the Task Coder agent to add appropriate print statements to the console to aid in debugging."
                "\n- Instruct the Task Coder agent to always check the content resulting from the creation, deletion, or modification of files performed within the Python script."
                "\n- Refer to the following code snippet for tasks involving an NLP model, such as summarization, content creation, or content analysis:"
                "\n```python"
                "\nfrom src.actions import Actions # This is a local class, do not install it with pip."
                "\nactions = Actions()"
                "\noutput = actions.inference('Summarize the following text: ' + content) # Ensure to concatenate variables if required."
                "\n```"
            )
        }

    def get_task_coder_system_prompt(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                "You are an extremely smart AI agent known as the Task Coder. Your role is to create only Python code based on the previous content and instructions."
                "\nYou shall only write Python code."
                "\nYou shall write the code inside a single code block."
                "\nYour message should start directly with ```python"
                "\n# Your Python code should be inside this code block."
                "\n```"
                "\nYou are not allowed to write comments, opinions, or explanations either before or after the code block."
            )
        }

    def get_task_analyzer_system_prompt(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                "You are an advanced AI agent named Task Analyzer AI. Your role is to evaluate the output from the code execution found inside the '[OUTPUT][/OUTPUT]' tags and determine if it meets the task requirements. "
                "If the output meets the task requirements, respond only with 'POSITIVE'; if it does not, respond only with 'NEGATIVE'."
            )
        }

    def truncate_left(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[-max_words:])
        return text

    def inference(self, agent: str, text: str, color: str, reset: str, max_tokens: int = 8192) -> str:
        max_words = int(max_tokens / (TOKEN_TO_WORD_RATIO * SAFETY_MARGIN))
        text = self.truncate_left(text, max_words)

        if agent == "builder":
            system = self.get_task_builder_system_prompt()
        elif agent == "instructor":
            system = self.get_task_instructor_system_prompt()
        elif agent == "coder":
            system = self.get_task_coder_system_prompt()
        elif agent == "analyzer":
            system = self.get_task_analyzer_system_prompt()

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

        print(f"{color}> ", end="")
        response = ""
        try:
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    response += content
        except Exception as e:
            print(f"Error: {e}")
        print(reset)

        return response.strip()

    def classify(self, input: str, threshold: int) -> bool:
        inputs = self.tokenizer(input, return_tensors="pt", truncation=True, padding=True, max_length=self.classifier_max_seq_length).to(self.device)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
        predicted_class_id = logits.argmax().item()
        return True if predicted_class_id >= threshold else False