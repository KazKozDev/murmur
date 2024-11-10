import pytest
from src.main import AgentOrchestrator

def test_orchestrator_creation():
    orchestrator = AgentOrchestrator()
    assert orchestrator is not None
    assert len(orchestrator.agents) == 4