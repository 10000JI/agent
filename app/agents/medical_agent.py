from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
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
from app.utils.logger import custom_logger


@dataclass
class ChatResponse:
    """ToolStrategy용 응답 스키마 (dataclass여야 ToolStrategy가 정상 동작)"""
    message_id: str
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


# 의료 에이전트에서 사용하는 도구 목록
MEDICAL_TOOLS = [
    search_medical_info, search_hospitals, get_drug_info,
    search_emergency_rooms, search_pharmacies,
]


def create_medical_agent(model: ChatOpenAI, checkpointer: MemorySaver):
    """의료 전문 에이전트를 생성합니다."""
    custom_logger.info("의료 에이전트 생성 중...")

    agent = create_agent(
        model=model,
        tools=MEDICAL_TOOLS,                    # 5개 도구 (병원, 약물, 응급실, 약국, ES 검색)
        system_prompt=MEDICAL_SYSTEM_PROMPT,     # 의료 도메인 역할 지시
        response_format=ToolStrategy(ChatResponse),            # 응답을 정해진 형식(message_id, content, metadata)으로 통일
        checkpointer=checkpointer,               # 대화 기록 저장 → thread_id로 멀티턴 대화 유지
        # middleware=[handle_tool_errors],  # ToolStrategy와 충돌 — agent 3에도 미사용
    )

    custom_logger.info(
        f"의료 에이전트 생성 완료 (도구: {[t.name for t in MEDICAL_TOOLS]})"
    )
    return agent
