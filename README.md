# Murmur

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![GitHub issues](https://img.shields.io/github/issues/KazKozDev/murmur)](https://github.com/KazKozDev/murmur/issues)
[![GitHub stars](https://img.shields.io/github/stars/KazKozDev/murmur)](https://github.com/KazKozDev/murmur/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/KazKozDev/murmur)](https://github.com/KazKozDev/murmur/network)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/KazKozDev/murmur)](https://github.com/KazKozDev/murmur/pulls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![aiohttp](https://img.shields.io/badge/aiohttp-3.8+-blue.svg)](https://docs.aiohttp.org/)

A sophisticated multi-agent system that orchestrates different specialized AI agents using local LLM models to process and respond to user queries. The system implements a pipeline of Interpreter, Reasoner, Generator, and Critic agents to provide well-thought-out and refined responses.

## 🌟 Features

- **Multi-Agent Architecture**: Four specialized agents working in concert:
  - **Interpreter**: Analyzes user intent and context
  - **Reasoner**: Develops logical approach to the problem
  - **Generator**: Creates initial responses
  - **Critic**: Reviews and refines generated content

- **Local LLM Integration**: Works with locally hosted language models through a REST API
- **Asynchronous Processing**: Built with `asyncio` for efficient concurrent operations
- **Robust Error Handling**: Comprehensive error management with retries and graceful fallbacks
- **Conversation Memory**: Maintains context through conversation history
- **Confidence Scoring**: Evaluates response quality with multiple metrics

## 🔧 Prerequisites

- Python 3.7+
- Local LLM server (compatible with Ollama API)
- Required Python packages:
  ```
  aiohttp
  asyncio
  ```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/KazKozDev/murmur.git
cd murmur
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your local LLM server is running (default: http://localhost:11434)

## 💻 Usage

1. Navigate to the project directory and run:
```bash
cd murmur
python src/main.py
```

Or navigate directly to the source directory:
```bash
cd murmur/src
python main.py
```

2. Enter your queries when prompted. Type 'quit' to exit.

Example interaction:
```python
Enter your message: What is the capital of France?

Response: The capital of France is Paris.
Confidence: 0.95
```

## 🏗️ Architecture

The system follows a pipeline architecture:

1. **User Input** → **Interpreter Agent**
   - Analyzes core intent and context
   - Identifies implicit requirements

2. **Interpreted Message** → **Reasoner Agent**
   - Breaks down the problem
   - Develops logical approach

3. **Reasoning** → **Generator Agent**
   - Creates initial response
   - Structures content clearly

4. **Generated Content** → **Critic Agent**
   - Reviews for accuracy and completeness
   - Suggests improvements
   - Produces final version

## ⚙️ Configuration

The system uses the following default models:
- Interpreter: mistral-nemo:latest
- Reasoner: llama3.2-vision:11b
- Generator: gemma2:9b
- Critic: llama3.2-vision:11b

Models can be configured by modifying the `AgentOrchestrator` initialization.

## 🔐 Error Handling

The system implements multiple layers of error handling:
- Connection retries (max 3 attempts)
- Timeout management
- Graceful degradation
- Comprehensive error logging

## 🤝 Contributing

1. Fork the repository (https://github.com/KazKozDev/murmur/fork)
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🪲 Known Issues

- High CPU usage with multiple concurrent requests
- Memory consumption may increase with long conversations
- Some LLM models may require significant local resources

## 🔜 Future Improvements

- [ ] Add support for streaming responses
- [ ] Implement agent personality customization
- [ ] Add websocket support for real-time communication
- [ ] Enhance conversation memory management
- [ ] Add support for more LLM providers
- [ ] Implement response caching

## 📞 Support

For support, please open an issue in the [GitHub repository](https://github.com/KazKozDev/murmur/issues) or contact the maintainers.

## 🙏 Acknowledgments

- Thanks to the Ollama team for their local LLM server
- Inspired by multi-agent architectures in AI systems