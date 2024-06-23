# AutoInterpreter

AutoInterpreter is a sophisticated AI agent designed to autonomously process and analyze tasks using Python and feedback analysis. The agent operates by generating, instructing, coding, analyzing, and evaluating tasks based on the provided objective, ensuring a seamless and intelligent automation experience. Additionally, the agent is capable of inferring the model on its own if it deems it appropriate.

## Warning
Please be cautious about the objective you give the agent, as it is capable of deleting or modifying files on its own and has full access to your PC.

## Disclaimer
AutoInterpreter was created as a side project and is currently unable to perform complex tasks. Feel free to submit pull requests if you want to improve the project. I take no responsibility for any harm caused by improper use by the user or by malicious or harmful actions performed by the agent.

## Getting Started

1. **Installation**:
   ```bash
   git clone https://github.com/motexture/AutoInterpreter
   cd AutoInterpreter
   python3 -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```
2. **Config file**:
   Open the config/config.yaml file and enter your GPT-4 api key and specify the model
   You can also use a local LLM as long as it uses the OpenAI functions

3. **Set objective**:
   Set the run->objective variable with your objective

4. **Run the project**
   ```bash
   python main.py
   ```

## Key Features

- **Automated Task Processing**: 
  - AutoInterpreter autonomously processes tasks by generating specific, detailed tasks, providing step-by-step instructions, coding, executing, and analyzing the results.

- **Model Integration**: 
  - Utilizes advanced machine learning models for inference, sentiment analysis, and natural language processing to ensure accurate and efficient task handling.

- **Memory Management**: 
  - Efficient memory system to store and retrieve task information and embeddings, ensuring continuity and relevance in task processing.

- **Customizable Configuration**: 
  - Easily adjustable parameters through the configuration file, allowing users to tailor the agent’s behavior to their specific needs.

- **Search and Scrape Functionalities**: 
  - Integrates Google Custom Search and web scraping capabilities to retrieve relevant information from the web.

- **Comprehensive Testing**: 
  - Includes a robust testing framework to validate the functionality and reliability of the system components.
