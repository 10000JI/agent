# Medical Agent Tools 리팩토링 설계

> 작성일: 2026-03-12
> 상태: 승인 대기

## 개요

피드백 5개 항목을 반영하여 `tools.py`, `medical_agent.py`, `agent_service.py`를 수정한다.
파일 구조 변경은 최소화하되, `region_codes.py` 1개 파일만 신규 추가한다.

## 변경 사항 요약

| # | 이슈 | 해결 방식 | 영향 파일 |
|---|------|-----------|-----------|
| 1 | `search_hospitals`에서 `emdongNm`에 모든 지역명 입력 → 상위 행정구역 검색 실패 | `_parse_region()` 유틸리티 도입, `sidoCd`/`sgguCd`/`emdongNm` 3단계 분류 | `tools.py`, `region_codes.py`(신규) |
| 2 | `search_emergency_rooms`에서 `STAGE2` 항상 빈 문자열 → 시군구 세분화 불가 | `_parse_region()`에서 `sido_name`/`sggu_name` 반환, STAGE2에 매핑 | `tools.py`, `region_codes.py`(신규) |
| 3 | 동기 `httpx.Client` → FastAPI 비동기 환경에서 이벤트 루프 블로킹 | `httpx.AsyncClient` + `async def` 도구로 전환 (4개 도구) | `tools.py` |
| 4 | `_get_specialty_code` 코드 번호 오차 (재활의학과 등 1칸씩 밀림) | API 호출 검증 완료, 코드 테이블 수정 + 직업환경의학과 추가 | `tools.py` |
| 5 | `create_react_agent()` → LangChain 1.0 `create_agent()` 마이그레이션 | `langchain.agents.create_agent` 사용, 노드명 "model"/"tools"로 통일 | `medical_agent.py`, `agent_service.py` |

## 상세 설계

### 1. 비동기 전환 (#3)

외부 API를 호출하는 4개 도구를 `async def`로 전환한다.

**변경 대상**: `search_hospitals`, `get_drug_info`, `search_emergency_rooms`, `search_pharmacies`

```python
# Before
@tool
def search_hospitals(region: str, specialty: Optional[str] = None) -> str:
    with httpx.Client(timeout=10) as client:
        response = client.get(url, params=params)

# After
@tool
async def search_hospitals(region: str, specialty: Optional[str] = None) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params=params)
```

**`search_medical_info`는 동기 유지**: `ElasticsearchRetriever.invoke()`가 동기 메서드이며, LangGraph는 동기 도구를 스레드풀에서 실행하므로 블로킹 문제 없음.

### 2. 지역 파라미터 통합 (#1, #2)

#### 2-1. `region_codes.py` 신규 파일

`app/agents/region_codes.py`에 전국 시도(17개) + 시군구(~250개) 매핑 데이터를 정의한다.

```python
# 구조
_REGION_DB = {
    # 시도 레벨
    "서울": {"sidoCd": "110000", "sgguCd": "", "sido_name": "서울특별시", "sggu_name": ""},
    "부산": {"sidoCd": "210000", "sgguCd": "", "sido_name": "부산광역시", "sggu_name": ""},
    ...
    # 시군구 레벨 — 고유한 이름 (강남구 등)
    "강남구": {"sidoCd": "110000", "sgguCd": "110023", "sido_name": "서울특별시", "sggu_name": "강남구"},
    ...
    # 중복 시군구 (중구, 남구, 북구, 동구, 서구 등) — "시도 시군구" 복합 키로 저장
    "서울 중구": {"sidoCd": "110000", "sgguCd": "110002", "sido_name": "서울특별시", "sggu_name": "중구"},
    "부산 중구": {"sidoCd": "210000", "sgguCd": "210001", "sido_name": "부산광역시", "sggu_name": "중구"},
    ...
}
```

#### 2-2. `parse_region()` 함수 — 중복 시군구 처리

```python
def parse_region(region: str) -> dict:
    """지역명을 파싱하여 API별 파라미터 반환.

    처리 순서:
    1. 원문 전체 매칭 ("서울 중구" → 복합 키 직접 매칭)
    2. 공백 분리 후 "시도 + 시군구" 복합 키 매칭 ("서울 강남구")
    3. 단독 키 매칭 ("강남구", "서울")
    4. 폴백 — 원문 그대로 반환
    """
```

**중복 시군구 disambiguiation**: `중구`, `남구`, `북구`, `동구`, `서구` 등 여러 도시에 존재하는 구 이름은 단독 키로 등록하지 않는다. 반드시 "시도 시군구" 형태로만 매칭된다. 에이전트 프롬프트에서 시도+시군구를 함께 넘기도록 유도한다.

#### 2-3. 각 도구에서의 사용

```python
# 병원 도구 — sidoCd/sgguCd 사용, emdongNm 비움
parsed = parse_region(region)
params = {
    "sidoCd": parsed["sidoCd"],
    "sgguCd": parsed["sgguCd"],
    "emdongNm": "",  # 코드 매칭 성공 시 비움
}

# 약국 도구 — 동일 구조 (기존 emdongNm만 사용 → sidoCd/sgguCd 추가)
parsed = parse_region(region)
params = {
    "sidoCd": parsed["sidoCd"],
    "sgguCd": parsed["sgguCd"],
    "emdongNm": "",
}

# 응급실 도구 — 한글 이름 사용
parsed = parse_region(region)
params = {
    "STAGE1": parsed["sido_name"],
    "STAGE2": parsed["sggu_name"],
}
```

#### 2-4. 폴백 처리

매핑에 없는 지역명은 `parse_region()`이 아래를 반환한다:
```python
{"sidoCd": "", "sgguCd": "", "sido_name": "", "sggu_name": "", "raw": region}
```

각 도구에서 `sidoCd`가 빈 문자열이면 기존 방식으로 폴백:
- 병원/약국: `emdongNm=region` (읍면동 검색)
- 응급실: `STAGE1=region` (원문 그대로)

### 3. 진료과목 코드 수정 (#4)

API 호출로 검증 완료된 결과를 반영한다.

| 진료과목 | 현재(잘못된) 코드 | 수정 후(검증된) 코드 |
|----------|-------------------|---------------------|
| 직업환경의학과 | (누락) | **20** |
| 재활의학과 | 20 | **21** |
| 핵의학과 | 21 | **22** |
| 가정의학과 | 22 | **23** |
| 응급의학과 | 23 | **24** |

### 4. `create_agent` 마이그레이션 (#5)

#### 4-1. `medical_agent.py`

```python
# Before
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm, tools=MEDICAL_TOOLS,
    prompt=MEDICAL_SYSTEM_PROMPT, checkpointer=checkpointer,
)

# After
from langchain.agents import create_agent

agent = create_agent(
    model=llm, tools=MEDICAL_TOOLS,
    system_prompt=MEDICAL_SYSTEM_PROMPT, checkpointer=checkpointer,
)
```

- `prompt` → `system_prompt` (파라미터명 변경)
- `response_format`은 `None` (기본값, 자유 텍스트 응답 유지)

#### 4-2. `agent_service.py` — 노드명 정리

`create_agent`의 노드명은 `"model"`, `"tools"` (검증 완료).
기존 `"agent"` 분기를 제거한다. whitelist와 라우팅 모두 수정.

```python
# Before (line 99 whitelist)
if not event or step not in ("agent", "model", "tools"):
    continue

# After
if not event or step not in ("model", "tools"):
    continue

# Before (line 107 routing)
if step in ("agent", "model"):

# After
if step == "model":
```

## 변경 파일 목록

| 파일 | 변경 유형 |
|------|-----------|
| `app/agents/tools.py` | 수정 — 비동기 전환, 지역 파라미터 변경, 진료과목 코드 수정 |
| `app/agents/region_codes.py` | **신규** — 전국 시도/시군구 코드 매핑 + `parse_region()` |
| `app/agents/medical_agent.py` | 수정 — `create_react_agent` → `create_agent` |
| `app/services/agent_service.py` | 수정 — 노드명 "agent" 분기 제거 |

## 변경하지 않는 것

- **UI 프로젝트** (`/Users/n-mjkim/IdeaProjects/ui/`): SSE 이벤트 포맷이 동일하므로 수정 불필요
- **`prompts.py`**: 시스템 프롬프트 변경 없음
- **`config.py`**: 설정 변경 없음
- **`pyproject.toml`**: 의존성 변경 없음 (`httpx`는 이미 설치됨)
