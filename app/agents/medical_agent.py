from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app.agents.prompts import MEDICAL_SYSTEM_PROMPT
from app.agents.tools import (
    search_medical_info, search_hospitals, get_drug_info,
    search_emergency_rooms, search_pharmacies,
)
from app.utils.logger import custom_logger


# 의료 에이전트에서 사용하는 도구 목록
MEDICAL_TOOLS = [
    search_medical_info, search_hospitals, get_drug_info,
    search_emergency_rooms, search_pharmacies,
]


def create_medical_agent(llm: ChatOpenAI, checkpointer: MemorySaver):
    """의료 전문 에이전트를 생성합니다.

    Args:
        llm: ChatOpenAI 모델 인스턴스
        checkpointer: 대화 기록 체크포인터 (MemorySaver 또는 SqliteSaver)

    Returns:
        LangGraph 에이전트 인스턴스
    """
    custom_logger.info("의료 에이전트 생성 중...")

    agent = create_react_agent(
        model=llm,
        tools=MEDICAL_TOOLS,
        prompt=MEDICAL_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    custom_logger.info(
        f"의료 에이전트 생성 완료 (도구: {[t.name for t in MEDICAL_TOOLS]})"
    )
    return agent
