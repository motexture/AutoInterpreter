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
   pip install -r requirements.txt
   ```
2. **Config file**:
   Open the config/config.yaml file and enter your GPT-4 api key and specify the model
   You can also use a local LLM as long as it uses the OpenAI functions
   Remember to replace python with python3 if your python env is set to python3

4. **Set objective**:
   Set the run->objective variable with your objective

5. **Run the project**
   ```bash
   python run.py
   ```
