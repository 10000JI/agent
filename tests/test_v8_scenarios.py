"""의료 AI 에이전트 시나리오 테스트.

실제 OpenAI API와 외부 도구를 호출하여 SSE 스트리밍 파이프라인의 다양한 시나리오를 검증한다.
- Case 1: 단일 도구 호출 (병원 검색)
- Case 2: 단일 도구 호출 (의약품 정보)
- Case 3: 멀티턴 대화 (동일 thread_id로 문맥 유지)
- Case 4: 도구 없이 직접 응답 (일반 인사)
- Case 5: 단일 도구 호출 (응급실 검색)
- Case 6: 단일 도구 호출 (의료 문서 검색)
- Case 7: 복합 질문 — 병원 + 약국
- Case 8: 증상 기반 진료과목 자동 추론
"""
import pytest
import json
import uuid
from fastapi.testclient import TestClient
from typing import List, Dict, Any


def parse_sse_response(response_text: str) -> List[Dict[str, Any]]:
    """SSE 응답을 파싱하는 헬퍼 함수"""
    events = []
    for line in response_text.strip().split('\n'):
        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                pass
    return events


def get_tool_calls(events: List[Dict[str, Any]]) -> List[str]:
    """SSE 이벤트에서 도구 호출 목록을 추출하는 헬퍼"""
    tool_calls = []
    for event in events:
        if event.get("step") == "model" and "tool_calls" in event:
            tool_calls.extend(event["tool_calls"])
    return tool_calls


def get_done_event(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """SSE 이벤트에서 최종 응답(done)을 추출하는 헬퍼"""
    done_events = [e for e in events if e.get("step") == "done"]
    assert len(done_events) >= 1, "No 'done' event found"
    return done_events[-1]


@pytest.mark.order(3)
def test_case1_hospital_search(client: TestClient):
    """
    Case 1: 단일 도구 호출 — 병원 검색
    사용자 질문: "강남구 정형외과 병원 추천해줘"
    기대: search_hospitals 호출, done 이벤트에 병원 정보 포함
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "강남구 정형외과 병원 추천해줘"
        }
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    assert "search_hospitals" in tool_calls, f"search_hospitals not found in {tool_calls}"
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0


@pytest.mark.order(4)
def test_case2_drug_info(client: TestClient):
    """
    Case 2: 단일 도구 호출 — 의약품 정보 조회
    사용자 질문: "아스피린 효능이랑 부작용 알려줘"
    기대: get_drug_info 호출, done 이벤트에 의약품 정보 포함
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "아스피린 효능이랑 부작용 알려줘"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    assert "get_drug_info" in tool_calls, f"get_drug_info not found in {tool_calls}"
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0


@pytest.mark.order(5)
def test_case3_multiturn_conversation(client: TestClient):
    """
    Case 3: 멀티턴 대화 (동일 thread_id로 문맥 유지)
    1차: "서울 응급실 빈 병상 알려줘"
    2차: "거기 근처 약국도 알려줘"
    기대: 1차에서 search_emergency_rooms, 2차에서 search_pharmacies 호출
          2차 응답이 서울 지역 맥락을 유지
    """
    thread_id = str(uuid.uuid4())

    # 1차 요청
    response1 = client.post(
        "/api/v1/chat",
        json={
            "thread_id": thread_id,
            "message": "서울 응급실 빈 병상 알려줘"
        }
    )

    assert response1.status_code == 200
    events1 = parse_sse_response(response1.text)
    tool_calls1 = get_tool_calls(events1)
    done1 = get_done_event(events1)

    assert "search_emergency_rooms" in tool_calls1
    assert len(done1["content"]) > 0

    # 2차 요청 — 동일 thread_id로 문맥 유지
    response2 = client.post(
        "/api/v1/chat",
        json={
            "thread_id": thread_id,
            "message": "거기 근처 약국도 알려줘"
        }
    )

    assert response2.status_code == 200
    events2 = parse_sse_response(response2.text)
    tool_calls2 = get_tool_calls(events2)
    done2 = get_done_event(events2)

    assert "search_pharmacies" in tool_calls2, f"search_pharmacies not found in {tool_calls2}"
    assert len(done2["content"]) > 0


@pytest.mark.order(6)
def test_case4_no_tool_greeting(client: TestClient):
    """
    Case 4: 도구 호출 없이 직접 응답 — 일반 인사
    사용자 질문: "안녕하세요"
    기대: 도구 호출 없이 done 이벤트만 반환
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "안녕하세요"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)

    # Planning 이벤트 제외하고 실제 도구 호출이 없어야 함
    tool_calls = get_tool_calls(events)
    actual_tools = [t for t in tool_calls if t != "Planning"]
    assert len(actual_tools) == 0, f"도구 호출 없이 응답해야 하는데 {actual_tools} 호출됨"

    done = get_done_event(events)
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0


@pytest.mark.order(7)
def test_case5_emergency_room(client: TestClient):
    """
    Case 5: 단일 도구 호출 — 응급실 실시간 정보
    사용자 질문: "부산 응급실 빈 병상 알려줘"
    기대: search_emergency_rooms 호출, done 이벤트에 응급실 정보 포함
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "부산 응급실 빈 병상 알려줘"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    assert "search_emergency_rooms" in tool_calls, f"search_emergency_rooms not found in {tool_calls}"
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0


@pytest.mark.order(8)
def test_case6_medical_info(client: TestClient):
    """
    Case 6: 단일 도구 호출 — 의료 문서 검색
    사용자 질문: "고혈압 관리법 알려줘"
    기대: search_medical_info 호출, done 이벤트에 의료 정보 포함
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "고혈압 관리법 알려줘"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    assert "search_medical_info" in tool_calls, f"search_medical_info not found in {tool_calls}"
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0


@pytest.mark.order(9)
def test_case7_hospital_and_pharmacy(client: TestClient):
    """
    Case 7: 복합 질문 — 병원 + 약국
    사용자 질문: "종로구 정형외과 병원이랑 근처 약국도 같이 알려줘"
    기대: search_hospitals + search_pharmacies 호출
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "종로구 정형외과 병원이랑 근처 약국도 같이 알려줘"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    assert "search_hospitals" in tool_calls, f"search_hospitals not found in {tool_calls}"
    assert "search_pharmacies" in tool_calls, f"search_pharmacies not found in {tool_calls}"
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0


@pytest.mark.order(10)
def test_case8_symptom_specialty_inference(client: TestClient):
    """
    Case 8: 증상 기반 진료과목 자동 추론
    사용자 질문: "무릎이 아프고 걷기 힘든데 강남구 병원 추천해줘"
    기대: search_hospitals 호출 (LLM이 정형외과를 자동 추론)
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": str(uuid.uuid4()),
            "message": "무릎이 아프고 걷기 힘든데 강남구 병원 추천해줘"
        }
    )

    assert response.status_code == 200

    events = parse_sse_response(response.text)
    tool_calls = get_tool_calls(events)
    done = get_done_event(events)

    assert "search_hospitals" in tool_calls, f"search_hospitals not found in {tool_calls}"
    assert done["role"] == "assistant"
    assert len(done["content"]) > 0
