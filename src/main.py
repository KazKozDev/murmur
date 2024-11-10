import asyncio
import aiohttp
import logging
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LocalLLMError(Exception):
    """Custom exception for local LLM server errors"""
    pass


class AgentRole(Enum):
    INTERPRETER = "interpreter"
    REASONER = "reasoner"
    GENERATOR = "generator"
    CRITIC = "critic"


@dataclass
class Message:
    role: str
    content: str
    metadata: Dict[str, Any] = None


class AgentResponse:
    def __init__(self, content: str, confidence: float, reasoning: str):
        self.content = content
        self.confidence = confidence
        self.reasoning = reasoning
        self.timestamp = time.time()


class LocalLLMConnector:
    """Handles communication with local LLM server"""
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self._session = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def call_model(self, model: str, prompt: str, 
                        max_retries: int = 3, timeout: float = 30.0) -> str:
        session = await self.get_session()

        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "temperature": 0.7,
                        "max_tokens": 2048,
                        "stream": False
                    },
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    elif response.status == 404:
                        raise LocalLLMError(f"Model {model} not found")
                    elif response.status == 503:
                        raise LocalLLMError("LLM server is unavailable")
                    else:
                        error_text = await response.text()
                        raise LocalLLMError(f"HTTP {response.status}: {error_text}")

            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    raise LocalLLMError("Request timed out")
                logging.warning(f"Timeout occurred. Retrying {attempt + 1}/{max_retries}...")
                await asyncio.sleep(1)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise LocalLLMError(f"Error calling model: {str(e)}")
                logging.warning(f"Error occurred: {str(e)}. Retrying {attempt + 1}/{max_retries}...")
                await asyncio.sleep(1)

        raise LocalLLMError("Max retries exceeded")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class BaseAgent:
    def __init__(self, name: str, model: str, role: AgentRole):
        self.name = name
        self.model = model
        self.role = role
        self.conversation_memory = deque(maxlen=10)
        self.llm_connector = LocalLLMConnector()

    async def process(self, message: Message, context: List[Message]) -> AgentResponse:
        self.conversation_memory.append(message)

        try:
            # Call the model with appropriate prompt based on role
            response = await self.llm_connector.call_model(
                self.model,
                self._create_prompt(message, context)
            )

            # Parse response and calculate confidence
            content, reasoning = self._parse_response(response)
            confidence = self._calculate_confidence(content, message)

            return AgentResponse(content, confidence, reasoning)

        except LocalLLMError as e:
            logging.error(f"LLM Error in {self.name}: {str(e)}")
            return AgentResponse(
                "I encountered an error processing your request. Please try again.",
                0.0,
                f"Error: {str(e)}"
            )
        except Exception as e:
            logging.error(f"General Error in {self.name}: {str(e)}")
            return AgentResponse(
                "An unexpected error occurred. Please try again.",
                0.0,
                f"Error: {str(e)}"
            )

    def _create_prompt(self, message: Message, context: List[Message]) -> str:
        base_prompt = {
            AgentRole.INTERPRETER: """
As an Interpreter agent, analyze the user's message to understand:
1. Core intent and goals
2. Important context and constraints
3. Implicit requirements

Message: {message}

Provide your understanding in a clear format with these sections:
- Intent:
- Key Points:
- Requirements:
""",
            AgentRole.REASONER: """
As a Reasoner agent, analyze the interpreted message and:
1. Break down the problem
2. Identify key considerations
3. Develop logical approach

Interpreted Message: {message}

Provide your reasoning in these sections:
- Analysis:
- Considerations:
- Approach:
""",
            AgentRole.GENERATOR: """
As a Generator agent, create a response that:
1. Directly addresses the user's needs
2. Incorporates the reasoning provided
3. Is clear and well-structured

Reasoning: {message}

Provide:
- Response:
- Explanation:
""",
            AgentRole.CRITIC: """
As a Critic agent, review the generated response for:
1. Accuracy and completeness
2. Clarity and coherence
3. Areas for improvement

Generated Response: {message}

Provide:
- Evaluation:
- Suggestions:
- Final Version:
"""
        }

        prompt_template = base_prompt[self.role]
        context_str = "\n".join(f"{m.role}: {m.content}" for m in context[-3:])

        return prompt_template.format(
            message=message.content,
            context=context_str
        )

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the structured response into content and reasoning"""
        try:
            # Split response into sections
            sections = response.split('\n')
            content_parts = []
            reasoning_parts = []

            for section in sections:
                if section.startswith(('Response:', 'Final Version:')):
                    content_parts.append(section.split(':', 1)[1].strip())
                elif section.startswith(('Analysis:', 'Reasoning:', 'Evaluation:')):
                    reasoning_parts.append(section.split(':', 1)[1].strip())

            content = ' '.join(content_parts) if content_parts else response
            reasoning = ' '.join(reasoning_parts) if reasoning_parts else "No explicit reasoning provided"

            return content, reasoning

        except Exception as e:
            logging.error(f"Error parsing response: {str(e)}")
            return response, "Error parsing structured response"

    def _calculate_confidence(self, content: str, message: Message) -> float:
        """Calculate confidence score for the response"""
        if not content:
            return 0.0

        # Basic confidence metrics
        metrics = {
            'length': min(len(content.split()) / 100, 1.0),
            'structure': 1.0 if any(section in content for section in ['Response:', 'Analysis:', 'Evaluation:']) else 0.5,
            'relevance': self._calculate_relevance(content, message.content)
        }

        return sum(metrics.values()) / len(metrics)

    def _calculate_relevance(self, response: str, query: str) -> float:
        """Calculate relevance score between response and query"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(response_words))
        return min(overlap / len(query_words), 1.0)


class AgentOrchestrator:
    def __init__(self):
        self.agents = {
            AgentRole.INTERPRETER: BaseAgent("Interpreter", "mistral-nemo:latest", AgentRole.INTERPRETER),
            AgentRole.REASONER: BaseAgent("Reasoner", "llama3.2-vision:11b", AgentRole.REASONER),
            AgentRole.GENERATOR: BaseAgent("Generator", "gemma2:9b", AgentRole.GENERATOR),
            AgentRole.CRITIC: BaseAgent("Critic", "llama3.2-vision:11b", AgentRole.CRITIC)
        }
        self.conversation_history = []

    async def process_message(self, user_message: str) -> Dict[str, Any]:
        message = Message(role="user", content=user_message)
        self.conversation_history.append(message)

        try:
            # Step 1: Interpret
            interpretation = await self.agents[AgentRole.INTERPRETER].process(
                message,
                self.conversation_history[-5:]
            )

            if interpretation.confidence < 0.5:
                return {
                    "response": "I'm having trouble understanding your message. Could you please rephrase it?",
                    "confidence": interpretation.confidence,
                    "error": None
                }

            # Steps 2-4: Reason, Generate, Critique
            reasoning = await self.agents[AgentRole.REASONER].process(
                Message("interpreter", interpretation.content),
                self.conversation_history[-5:]
            )

            generation = await self.agents[AgentRole.GENERATOR].process(
                Message("reasoner", reasoning.content),
                self.conversation_history[-5:]
            )

            critique = await self.agents[AgentRole.CRITIC].process(
                Message("generator", generation.content),
                self.conversation_history[-5:]
            )

            # Use critic's improved version if confidence is higher
            final_response = (
                critique.content if critique.confidence > generation.confidence
                else generation.content
            )

            response = {
                "response": final_response,
                "confidence": max(critique.confidence, generation.confidence),
                "interpretation": interpretation.content,
                "reasoning": reasoning.content,
                "generation": generation.content,
                "critique": critique.content,
                "error": None
            }

            self.conversation_history.append(Message("assistant", final_response))
            return response

        except LocalLLMError as e:
            error_msg = f"Error connecting to LLM server: {str(e)}"
            logging.error(error_msg)
            return {
                "response": "I'm currently unable to process requests due to a technical issue. Please try again later.",
                "confidence": 0.0,
                "error": error_msg
            }

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logging.error(error_msg)
            return {
                "response": "An unexpected error occurred. Please try again.",
                "confidence": 0.0,
                "error": error_msg
            }

    async def cleanup(self):
        """Clean up resources"""
        for agent in self.agents.values():
            await agent.llm_connector.close()


async def main():
    orchestrator = AgentOrchestrator()
    
    try:
        while True:
            user_input = input("\nEnter your message (or 'quit' to exit): ")
            
            if user_input.lower() == 'quit':
                break
                
            result = await orchestrator.process_message(user_input)
            
            print(f"\nResponse: {result['response']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            if result['error']:
                print(f"Error: {result['error']}")
                
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())