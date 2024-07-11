# Mix of Agents MurMur

MoA MurMur is an advanced framework utilizing large language models (LLMs) combined with an orchestration system. This sophisticated chat application leverages multiple AI models for enhanced conversational experiences.


## Key Features

- **Multi-Agent Architecture**: Utilizes a layered approach with multiple AI models for comprehensive and diverse responses. Models include:
  
  - Layer 1: Mistral and LLaVA:13B
  - Layer 2: LLaMA3:8B and Qwen2:7B
  - Final Layer: Gemma2:9B
    
- **Orchestration System**: Intelligently manages and coordinates multiple LLMs to produce cohesive and high-quality outputs.
- **Dynamic Model Selection**: Intelligently selects appropriate models based on the nature of user queries.
- **Advanced UI**: Built with Tkinter, featuring a customizable interface with adjustable font sizes and preference settings.
- **Image Analysis**: Supports image uploads and analysis using the LLaVA 13B model.
- **Web Search Integration**: Incorporates DuckDuckGo, Wikipedia, and Yahoo search capabilities for up-to-date information.
- **Response Quality Assurance**: Implements a sophisticated evaluation system to ensure high-quality outputs.
- **Caching Mechanism**: Employs a response cache to improve efficiency and reduce redundant API calls.
- **Asynchronous Processing**: Utilizes asyncio for non-blocking operations and improved responsiveness.


## Technical Details

- **Language**: Python
- **UI Framework**: Tkinter
- **AI Integration**: Uses Ollama for local AI model hosting and interactions
- **External Libraries**: Pillow, BeautifulSoup, scikit-learn, markdown, and more
- **Asynchronous Programming**: Leverages asyncio and aiohttp for efficient concurrent operations


## Prerequisites

- Python 3.x
- Ollama: This project requires Ollama to be installed and running. Ollama is used to run the AI models locally. You can find installation instructions and more information about Ollama at [https://ollama.ai/](https://ollama.ai/)


## Getting Started

1. Ensure you have Python installed on your system.
2. Install and start the Ollama server. Make sure it's running on http://localhost:11434 (default port).
3. Install required dependencies:
   
   ```
   pip install tkinter pillow aiohttp beautifulsoup4 scikit-learn markdown
   ```
  
5. Run the main script to launch the application.

## Usage

- Start a conversation by typing in the input field and pressing Enter or clicking Send.
- Upload images for analysis using the "Upload File" button.
- Adjust font size and user preferences as needed.
- Utilize web search integrations by enabling the respective checkboxes.

## Contributing

Contributions to enhance functionality, improve model integration, or optimize performance are welcome. Please submit pull requests or open issues for any bugs or feature requests.

## License

This 
