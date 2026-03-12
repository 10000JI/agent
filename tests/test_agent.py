"""에이전트 통합 테스트 (LLM + 도구).

실제 OpenAI API와 외부 도구를 호출하여 에이전트의 도구 선택과 응답을 검증한다.
"""
import pytest
from langchain_core.messages import HumanMessage


async def _run_agent(agent, question: str, thread_id: str):
    """에이전트 실행 후 도구 호출 목록과 최종 응답을 반환하는 헬퍼.

    Returns:
        (tool_names: list[str], final_content: str)
    """
    tool_names = []
    final_content = ""

    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        for step, event in chunk.items():
            if not event or step not in ("model", "tools"):
                continue
            messages = event.get("messages", [])
            if not messages:
                continue
            message = messages[0]

            if step == "model":
                if message.tool_calls:
                    tool_names.extend(t["name"] for t in message.tool_calls)
                elif message.content:
                    final_content = message.content

    return tool_names, final_content


class TestAgentIntegration:
    """에이전트가 올바른 도구를 선택하고 응답을 생성하는지 검증"""

    @pytest.mark.asyncio
    async def test_hospital_search(self, agent, thread_id):
        """'강남구 내과 병원 추천해줘' → search_hospitals 호출"""
        tools, content = await _run_agent(agent, "강남구 내과 병원 추천해줘", thread_id)
        assert "search_hospitals" in tools
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_drug_info(self, agent, thread_id):
        """'타이레놀 부작용 알려줘' → get_drug_info 호출"""
        tools, content = await _run_agent(agent, "타이레놀 부작용 알려줘", thread_id)
        assert "get_drug_info" in tools
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_emergency_room(self, agent, thread_id):
        """'서울 응급실 빈 병상 알려줘' → search_emergency_rooms 호출"""
        tools, content = await _run_agent(agent, "서울 응급실 빈 병상 알려줘", thread_id)
        assert "search_emergency_rooms" in tools
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_medical_info(self, agent, thread_id):
        """'결핵 치료 방법 알려줘' → search_medical_info 호출"""
        tools, content = await _run_agent(agent, "결핵 치료 방법 알려줘", thread_id)
        assert "search_medical_info" in tools
        assert len(content) > 0
