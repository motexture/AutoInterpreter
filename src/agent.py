import os
import yaml

from .models import Models
from .memory import Memory
from .utils import extract_code_blocks, execute_code

# Colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

class Agent:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), '../config/config.yaml'), 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.objective = self.config['run']['objective']
        self.input = ""
        self.output = ""

        self.memory = Memory(path=self.config['database']['path'])
        self.models = Models(self.config)

    def run(self):
        self.memory._initialize_db()
        
        while True:
            # Set input for Task Builder
            self.input = f"{self.output}\nObjective: {self.objective}\n\n"

            # Get output and generate embeddings
            self.task = self.models.inference("builder", self.input, BLUE, RESET, self.config['model']['model_context_length'])

            # Set input for Task Instructor
            self.input = f"Task: {self.task}\n\n"

            # Get output and generate embeddings
            self.instructions = self.models.inference("instructor", self.input, GREEN, RESET, self.config['model']['model_context_length'])
            self.output = f"{self.input}\nInstructions: {self.instructions}\nEnd of instructions section\n\n"

            # Task loop
            finished = False
            while True:
                self.input = f"{self.memory.remember(self.config['run']['memories_to_remember'])}\n{self.output}"

                # Get code output
                self.code = extract_code_blocks(self.models.inference("coder", self.input, YELLOW, RESET, self.config['model']['model_context_length']))
                
                # Execute code and print it
                print(f"> ", end="")
                self.output = execute_code(self.code, self.config['run']['python_env'], self.config['run']['code_execution_timeout']).strip()
                print()

                if self.output == "" or len(self.output) < 32:
                    self.output = "Code execution failed\n"

                # Prepare input
                self.input = f"Task: {self.task}\nInstructions: {self.instructions}\nCode: {self.code}\nCode output: {self.output}\nEnd of code output section.\n\n"
                

                # Get analysis
                self.analysis = self.models.inference("analyzer", self.input, MAGENTA, RESET, self.config['model']['model_context_length'])

                # Get output and generate embeddings
                self.output = f"{self.input}\nAnalysis: {self.analysis}\nEnd of analysis section.\n\n"

                # Save to memory
                self.memory.memorize(self.output)

                # Analyze output
                if self.models.classify(self.analysis, 3):
                    finished = True
                    break

                # Get output
                self.instructions = self.models.inference("instructor", self.output, GREEN, RESET, self.config['model']['model_context_length'])
                self.output = self.output + "\nInstruction: " + self.instructions + "\nEnd of instructions section.\n\n"
            # Finish program
            if finished:
                break
