import torch
import ssl

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from sentence_transformers import SentenceTransformer

ssl._create_default_https_context = ssl._create_unverified_context

TOKEN_TO_WORD_RATIO = 1.33  # Approximate ratio of tokens to words for English text
SAFETY_MARGIN = 1.2  # Safety margin to ensure we stay within the token limit
    
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
                "You are an efficient AI agent known as the Task Builder. Your role is to explain to the Task Instructor Agent what the objective is and what the user wants without giving instructions on how to complete it."
                "\n\nKey points to consider:"
                "\n- Answer only with the task description, avoiding any comments or additional text."
                "\n- Ensure the explanation is concise and easy to understand."
                "\n- Avoid giving instructions on how to solve it."
                "\n\nYour objective is to generate a clear, one-paragraph explanation that effectively communicates the objective and user requirements to the Task Instructor Agent. Ensure the explanation is straightforward and free of unnecessary information."
            )
        }
    
    def get_task_instructor_system_prompt(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                "You are an efficient AI agent known as the Task Instructor. Your role is to provide exhaustive step-by-step text instructions to help a Task Coder agent create Python code that will be executed immediately to complete a specified task."
                "\n\nYour instructions should never:"
                "\n- Include actions that require an API key unless the objective explicitly specifies it."
                "\n- Require human intervention."
                "\n- Create file(s) unless requested by the task; otherwise, provide the Python code without creating any file(s)."
                "\n- After writing the step-by-step instructions, do not provide the complete code at the end."
                "\n\nYour instructions should always:"
                "\n- Ensure all actions can be executed within a Python script."
                "\n- Provide detailed explanations of each step to ensure clarity and comprehension."
                "\n- Ensure that the script prints appropriate outputs to the console to aid in debugging the code."
                "\n- Ensure that any file creation, deletion, or modification is performed within the Python script."
                "\n- Clearly instruct the Task Coder agent to add print statements for each step and ensure that files created have content and are not empty."
                "\n- Refer to the following code snippet for tasks involving an NLP model, such as summarization, content creation, or content analysis:"
                "\n```python"
                "\nfrom src.actions import Actions # This is a local class, do not install it with pip"
                "\nactions = Actions()"
                "\noutput = actions.inference('Summarize the following text: ' + content) # Ensure to concatenate variables if required"
                "\n```"
                "\n- Provide detailed instructions for installing necessary packages with commands like `os.system('pip3 install requests --quiet')` before the imports."
                "\n- Be comprehensive and thorough, covering every necessary detail."
                "\n- Be precise and unambiguous, leaving no room for interpretation."
                "\n- Be organized in a logical sequence, ensuring a smooth workflow."
                "\n- Be actionable and direct, enabling the Task Coder to execute without additional clarifications."
            )
        }

    def get_task_coder_system_prompt(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                "You are an efficient AI agent known as the Task Coder. Your role is to create a detailed and accurate Python code snippet based on the step-by-step text instructions provided."
                "\n\nKey points to consider:"
                "\n- Follow the provided instructions precisely to create the Python code."
                "\n- Ensure the code is complete and ready for execution without any additional modifications."
                "\n- Ensure at all costs that the code is wrapped inside a code block like in this example: ```python\nprint('Hello World!')\n```"
                "\n- Include all necessary imports, function definitions, and logic as required by the instructions."
                "\n- The code must be well-structured, maintainable, and adhere to best practices."
                "\n- Include comments explaining what is happening at each step."
                "\n- Respond only with the code snippet and nothing else."
                "\n- The code should be uninterrupted, written in Python, and presented as a single block. If code in another language is required, it should be runnable or saveable to a file in that language using Python."
                "\n- Install all necessary packages with commands like `os.system('pip3 install requests --quiet')` before the imports."
                "\n- Never include print statements about statuses or actions; only print the variables of the results without additional text or comments."
                "\n\nYour objective is to precisely follow the instructions given without deviation, ensuring the code is correct and immediately executable."
            )
        }

    def get_task_analyzer_system_prompt(self) -> Dict[str, str]:
        return {
            "role": "system",
            "content": (
                "You are an efficient AI agent known as the Task Analyzer. Your primary role is to critically analyze the output from the last code execution to determine if the task has been successfully completed."
                "\n\nYou should, exactly in this order, follow the following steps until the statement becomes true:"
                "\n1. Check if the output after the last 'Code output:' section is not empty. If it is empty, first ensure that the code is correctly wrapped inside a code block. If it is, give instructions to fix it and highly criticize the Task Instructor agent for giving bad instructions, and analyze what could have caused the issue."
                "\n2. Check if the output after the last 'Code output:' section perfectly aligns with the initial request found after the 'Task:' section. If it doesn't align, then harshly criticize the Task Instructor agent and give it feedback about how to fix the issue."
                "\n3. If you are sure that the output after the last 'Code output:' section is valid, criticize in a fairly negative tone."
                "\n4. If unsure about the output after the last 'Code output:' section because the output is incomplete or anything like that, still highly criticize the Task Instructor agent."
                "\n5. If the instructions require the creation of external file(s) after the last 'Code output:' section but no debugging print statements have been performed to check if the file(s) were created, highly criticize in a fairly negative tone the Task Instructor agent."
                "\n6. If multiple import errors are present, make suggestions on how to fix them."
                "\n7. If the output aligns with the 'Task:' request and there are no errors in the output, praise the Task Instructor agent immensely and act extremely happy."
                "\n\nKey points to consider:"
                "\n- The positive and negative criticism is required because a Sentiment Analysis agent needs to understand if the task has been successfully completed or not. Take this into consideration.",
                "\n- Please conclude your analysis by clearly stating whether the findings are positive or negative. Use the format: FINAL REPORT: POSITIVE or FINAL REPORT: NEGATIVE."
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
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    response += chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error: {e}")
        print(reset)

        response = response.strip()
        
        return response
    
    def classify(self, input: str, treshold: int) -> bool:
        inputs = self.tokenizer(input, return_tensors="pt", truncation=True, padding=True, max_length=self.classifier_max_seq_length).to(self.device)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
        predicted_class_id = logits.argmax().item()
        return True if predicted_class_id >= treshold else False
