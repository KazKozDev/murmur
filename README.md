<div align="center">
  <img src="https://github.com/user-attachments/assets/dd45a96e-a809-44da-b51f-421fdc3302f0" width="180">
</div>

# Murmur 

*A Mix of Agents Orchestration System for Distributed LLM Processing*
<div align="left">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Code%20Style-Black-black" alt="Code Style: Black">
  <img src="https://img.shields.io/badge/Linting-flake8-blue" alt="Linting: flake8">
  <img src="https://img.shields.io/badge/Type%20Checking-mypy-blue" alt="Type Checking: mypy">
</div>
<br>
Murmur is an intelligent orchestration system built to integrate multiple large language models (LLMs) into a cohesive multi-agent pipeline. By coordinating agents specialized in interpreting, reasoning, generating, and critiquing responses, Murmur enables distributed task processing that leverages the strengths of models like mistral-nemo:latest, llama3.2-vision:11b, and gemma2:9b to deliver robust and confident answers to user input. The project is designed for environments where local LLM serving is preferred, ensuring low latency and fine-grained process control.  
<br><br>

The system is implemented in Python using asynchronous programming patterns with aiohttp and asyncio, ensuring that each request is handled efficiently. Murmur’s modular architecture makes it an good choice for developers looking to experiment with LLM orchestration, test different agent configurations, or deploy a local solution for conversational AI tasks.

## Features  
- **Multi-Agent Pipeline:**  
  - **Interpreter:** Analyzes user messages to extract intent, key points, and requirements.  
  - **Reasoner:** Breaks down the interpreted message, identifies critical factors, and formulates a logical approach.  
  - **Generator:** Creates well-structured responses directly addressing the user’s needs based on provided reasoning.  
  - **Critic:** Reviews and refines the generated response for accuracy, clarity, and completeness.  
- **Asynchronous Processing:** Built with Python’s asyncio and aiohttp for efficient, non-blocking request handling.  
- **Local LLM Integration:** Communicates with a local LLM server to ensure rapid processing and low latency.  
- **Code Quality Tools:** Integrated with Black, isort, flake8, and mypy to enforce coding standards and static type checking.

## Requirements  
- **Python:** Version 3.8 or higher  
- **Libraries:**  
  - aiohttp (for asynchronous HTTP requests)  
  - python-dotenv (for environment configuration)  
  - pytest, pytest-asyncio, pytest-aiohttp (for testing)  
  - Black, isort, flake8, mypy (for linting, formatting, and type checking)  
- **Local LLM Server:** Running at the default URL (http://localhost:11434) or a configurable endpoint as specified in the environment.  

## Installation  
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/KazKozDev/murmur.git
   cd murmur
   ```

2. **Create and Activate a Virtual Environment:**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

## Configuration  
Murmur allows for flexible configuration to suit various deployment environments:  

- **LLM Server URL:**  
  The base URL for the local LLM server is configurable via environment variables. By default, it is set to `http://localhost:11434`.  
  You can modify the URL in the `.env` file to point to your desired LLM service.

- **Logging:**  
  Logging is configured in the source code (see `src/main.py`) and prints INFO level logs to help track system behavior and debug potential issues.

- **Agent Prompts:**  
  Each agent (Interpreter, Reasoner, Generator, Critic) uses a prompt template defined in the code. You can modify these templates in `src/main.py` within the `_create_prompt` methods to customize agent behavior.

## Usage  
To run Murmur and start interacting with the multi-agent system, use the following instructions:

1. **Start the Application:**  
   ```bash
   python src/main.py
   ```

2. **Interact with the System:**  
   Once running, you will see a welcome message similar to:  
   “Mix of Agents Murmur v1.1.0: An intelligent orchestration system integrating multiple LLMs built on Ollama architecture.”  
   You can then enter your messages directly in the terminal. The system will process your input by sequentially routing it through the interpreter, reasoner, generator, and critic agents, eventually printing a detailed response along with a confidence score.

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)


