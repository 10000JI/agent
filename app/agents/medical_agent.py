from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from app.agents.middleware import handle_tool_errors
from app.agents.prompts import MEDICAL_SYSTEM_PROMPT
from app.agents.search_agent import search_medical_info
from app.agents.tools import (
    search_hospitals, get_drug_info,
    search_emergency_rooms, search_pharmacies,
)
from app.core.config import settings
from app.models.chat import ChatResponse
from app.utils.logger import custom_logger


# 의료 에이전트에서 사용하는 도구 목록
MEDICAL_TOOLS = [
    search_medical_info, search_hospitals, get_drug_info,
    search_emergency_rooms, search_pharmacies,
]


def create_medical_agent(checkpointer: MemorySaver):
    """의료 전문 에이전트를 생성합니다.

    create_agent는 내부적으로 LangGraph StateGraph를 구성하며,
    ToolNode의 기본 에러 핸들러가 도구 실행 예외를 자동으로 처리합니다.

    Args:
        checkpointer: 대화 기록 체크포인터 (MemorySaver 또는 SqliteSaver)

    Returns:
        LangGraph 에이전트 인스턴스
    """
    custom_logger.info("의료 에이전트 생성 중...")

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        streaming=True,
    )

    agent = create_agent(
        model=llm,                              # LLM (GPT-4.1)
        tools=MEDICAL_TOOLS,                    # 5개 도구 (병원, 약물, 응급실, 약국, ES 검색)
        system_prompt=MEDICAL_SYSTEM_PROMPT,     # 의료 도메인 역할 지시
        response_format=ChatResponse,            # 응답을 정해진 형식(message_id, content, metadata)으로 통일
        checkpointer=checkpointer,               # 대화 기록 저장 → thread_id로 멀티턴 대화 유지
        middleware=[handle_tool_errors],          # 도구 실행 중 에러 발생 시 자동 복구
    )

    custom_logger.info(
        f"의료 에이전트 생성 완료 (도구: {[t.name for t in MEDICAL_TOOLS]})"
    )
    return agent
