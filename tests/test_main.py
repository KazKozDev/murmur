import pytest
from src.main import AgentOrchestrator, Message, AgentRole

# Test Orchestrator Creation
def test_orchestrator_creation():
    orchestrator = AgentOrchestrator()
    assert orchestrator is not None
    assert len(orchestrator.agents) == 4
    assert orchestrator.conversation_history is not None

# Test Message Processing
@pytest.mark.asyncio
async def test_basic_message_processing():
    orchestrator = AgentOrchestrator()
    message = "Hello, this is a test message"
    
    result = await orchestrator.process_message(message)
    
    assert "response" in result
    assert "confidence" in result
    assert "error" in result
    assert result["error"] is None

# Test Agent Roles
def test_agent_roles():
    orchestrator = AgentOrchestrator()
    agent_roles = [agent.role for agent in orchestrator.agents.values()]
    
    assert AgentRole.INTERPRETER in agent_roles
    assert AgentRole.REASONER in agent_roles
    assert AgentRole.GENERATOR in agent_roles
    assert AgentRole.CRITIC in agent_roles

# Test Conversation History
@pytest.mark.asyncio
async def test_conversation_history():
    orchestrator = AgentOrchestrator()
    test_message = "Test message"
    
    initial_history_length = len(orchestrator.conversation_history)
    await orchestrator.process_message(test_message)
    
    assert len(orchestrator.conversation_history) > initial_history_length