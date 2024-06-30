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
        try:
            os.remove("debug.txt")
        except:
            pass

        # Init database
        self.memory._initialize_db()
        
        # Set input for Task Builder
        self.input = self.objective

        # Get output and generate embeddings
        self.task = self.models.inference("builder", self.input, BLUE, RESET, self.config['model']['model_context_length'])

        # Get output and generate embeddings
        self.instructions = self.models.inference("instructor", self.task, GREEN, RESET, self.config['model']['model_context_length'])

        # Prepare input  
        self.input = f"{self.task}\n\n{self.instructions}"

        # Task loop
        while True:
            # Extract code block
            self.code_input = extract_code_blocks(self.models.inference("coder", self.input, YELLOW, RESET, self.config['model']['model_context_length'])).strip()

            # Execute code block
            self.code_output = execute_code(self.code_input, self.config['run']['python_env'], self.config['run']['code_execution_timeout'])
            
            # Prepare input
            self.input = f"{self.input}\n\n{self.code_input}\n\n{self.code_output}"

            # Get analysis
            self.analysis = f"{self.task}\n\n{self.code_output}"
            self.analysis = self.models.inference('analyzer', self.analysis, MAGENTA, RESET, self.config['model']['model_context_length'])

            # Finalize and prepare input
            self.input = f"{self.input}\n\n{self.analysis}\n\n"

            try:
                with open('debug.txt', 'a') as file:
                    file.write(self.input)
            except:
                pass

            # Save to memory
            self.memory.memorize(self.input)

            # Analyze output
            if self.models.classify(self.analysis, 3):
                break
            
            # Remember memories
            self.input = f"{self.memory.remember(self.config['run']['memories_to_remember'])}\n\n{self.task}"

            # Set instructrions
            self.instructions = self.models.inference('instructor', self.input, GREEN, RESET, self.config['model']['model_context_length'])

            # Set input for next run
            self.input = f"{self.input}\n\n{self.instructions}"