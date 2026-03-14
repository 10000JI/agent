import pytest
import uuid
from fastapi.testclient import TestClient
from app.main import app

from langgraph.checkpoint.memory import MemorySaver

from app.agents.medical_agent import create_medical_agent


@pytest.fixture
def client():
    """FastAPI 테스트 클라이언트 fixture"""
    return TestClient(app)


@pytest.fixture
def thread_id():
    """테스트용 thread_id 생성"""
    return str(uuid.uuid4())


@pytest.fixture
def agent():
    """의료 에이전트 인스턴스"""
    checkpointer = MemorySaver()
    return create_medical_agent(checkpointer=checkpointer)
