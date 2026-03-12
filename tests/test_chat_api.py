"""SSE 엔드포인트 전체 파이프라인 테스트.

복합 질문으로 /api/v1/chat 엔드포인트의 SSE 스트리밍, 도구 호출, 최종 응답을 검증한다.
"""
import json
import uuid
import pytest
from fastapi.testclient import TestClient


class TestChatSSE:
    """POST /api/v1/chat SSE 스트리밍 테스트"""

    def test_complex_multi_tool_question(self, client: TestClient):
        """복합 질문 → 2개 도구 호출 + 최종 응답 포함 SSE 스트림

        질문: '두통이 심한데 타이레놀 먹어도 돼? 강남구 내과 병원도 알려줘'
        기대: get_drug_info + search_hospitals 호출, done 이벤트에 content 포함
        """
        response = client.post(
            "/api/v1/chat",
            json={
                "thread_id": str(uuid.uuid4()),
                "message": "두통이 심한데 타이레놀 먹어도 돼? 강남구 내과 병원도 알려줘",
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # SSE 이벤트 파싱
        events = []
        for line in response.text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    continue

        # 도구 호출 확인 — model step에서 tool_calls 수집
        all_tool_calls = []
        for event in events:
            if event.get("step") == "model" and "tool_calls" in event:
                all_tool_calls.extend(event["tool_calls"])

        assert "get_drug_info" in all_tool_calls, f"get_drug_info not found in {all_tool_calls}"
        assert "search_hospitals" in all_tool_calls, f"search_hospitals not found in {all_tool_calls}"

        # done 이벤트 확인
        done_events = [e for e in events if e.get("step") == "done"]
        assert len(done_events) >= 1, "No 'done' event found"

        done = done_events[-1]
        assert "message_id" in done
        assert "role" in done
        assert "content" in done
        assert "created_at" in done
        assert done["role"] == "assistant"
        assert len(done["content"]) > 0
