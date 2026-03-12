# Medical AI Agent

LangChain + LangGraph 기반의 **의료 전문 AI 에이전트** 서버입니다.
ReAct(Reasoning and Acting) 패턴을 활용하여 사용자의 의료 관련 질문에 대해 도구(Tool)를 자율적으로 선택하고 실행합니다.

Elasticsearch BM25 검색으로 38,000건 이상의 의료 문서를 검색하고, 공공데이터포털 REST API를 통해 병원/의약품/응급실/약국 정보를 실시간으로 조회합니다.

## 기술 스택

| 분류 | 기술 |
|------|------|
| **Backend Framework** | FastAPI 0.104+ |
| **Agent Framework** | LangChain v1.2, LangGraph (`create_agent`) |
| **LLM** | OpenAI GPT-4.1 (`langchain-openai`) |
| **검색 엔진** | Elasticsearch 8.x (BM25 기반 Full-text Search) |
| **ES 연동** | `langchain-elasticsearch` (`ElasticsearchRetriever`) |
| **외부 API** | 공공데이터포털 REST API (4개 서비스) |
| **HTTP Client** | httpx (비동기) |
| **설정 관리** | pydantic-settings (`.env` 기반) |
| **패키지 관리** | uv |
| **Python** | 3.11 ~ 3.13 |

## 아키텍처

```
┌─────────────┐     SSE Stream      ┌──────────────────────┐
│  React UI   │ ◄──────────────────► │  FastAPI Server      │
│ (port 5173) │   POST /api/v1/chat  │  (port 8000)         │
└─────────────┘                      │                      │
                                     │  ┌────────────────┐  │
                                     │  │ AgentService    │  │
                                     │  │ (SSE 스트리밍)   │  │
                                     │  └───────┬────────┘  │
                                     │          │           │
                                     │  ┌───────▼────────┐  │
                                     │  │ ReAct Agent     │  │
                                     │  │ (LangGraph)     │  │
                                     │  │                 │  │
                                     │  │ ┌─────────────┐ │  │
                                     │  │ │ GPT-4.1 LLM │ │  │
                                     │  │ └──────┬──────┘ │  │
                                     │  │        │        │  │
                                     │  │   Tool 선택/실행  │  │
                                     │  └───┬──┬──┬──┬──┬─┘  │
                                     └─────┼──┼──┼──┼──┼─────┘
                                           │  │  │  │  │
                              ┌────────────┘  │  │  │  └────────────┐
                              │               │  │  │               │
                         ┌────▼────┐   ┌──────▼──▼──────┐   ┌──────▼──────┐
                         │   ES    │   │  공공데이터포털   │   │ 국립중앙    │
                         │ BM25    │   │  (병원/의약품/   │   │ 의료원      │
                         │ 검색    │   │   약국)          │   │ (응급실)    │
                         └─────────┘   └─────────────────┘   └─────────────┘
```

### 요청-응답 흐름

1. 사용자가 React UI에서 메시지를 전송합니다.
2. `POST /api/v1/chat`으로 `{ thread_id, message }` 형태의 요청이 전달됩니다.
3. `AgentService`가 ReAct 에이전트를 생성하고, `astream(stream_mode="updates")`으로 스트리밍을 시작합니다.
4. 에이전트는 LLM이 판단한 도구를 실행하고, 각 단계를 SSE(Server-Sent Events)로 실시간 전달합니다.
5. 최종 응답이 생성되면 `"step": "done"` 이벤트로 전달됩니다.

### SSE 스트림 이벤트 포맷

```jsonc
// 1) 도구 호출 시작
{"step": "model", "tool_calls": ["search_medical_info"]}

// 2) 도구 실행 결과
{"step": "tools", "name": "search_medical_info", "content": "...검색 결과..."}

// 3) 최종 응답
{"step": "done", "message_id": "uuid", "role": "assistant", "content": "답변 내용", "metadata": {}, "created_at": "..."}
```

## 에이전트 도구 (Tools)

에이전트는 사용자 질문을 분석하여 아래 5개 도구 중 적절한 것을 자동으로 선택합니다.

### Tool 1: `search_medical_info` - 의료 문서 검색

| 항목 | 내용 |
|------|------|
| **데이터 출처** | Elasticsearch (`edu-collection` 인덱스, 38,154건) |
| **검색 방식** | BM25 Full-text Search (`ElasticsearchRetriever`) |
| **입력** | `query`: 증상, 질병명, 치료법 등 (예: `"결핵 치료"`, `"천식 증상"`) |
| **출력** | 상위 5개 문서 (출처, 연도, 본문 500자) |
| **문서 출처** | WHO, wikidoc, KSEM, 대한간학회, 대한결핵및호흡기학회 등 |

### Tool 2: `search_hospitals` - 병원 검색

| 항목 | 내용 |
|------|------|
| **데이터 출처** | 건강보험심사평가원 병원정보서비스 API |
| **입력** | `region`: 지역명 (필수), `specialty`: 진료과목 (선택) |
| **출력** | 병원명, 종류, 주소, 전화번호 (최대 5건) |
| **검색 범위** | 시도 → 시군구 → 읍면동 (예: `"서울"`, `"강남구"`, `"서울 중랑구 중화동"`) |
| **진료과목 매핑** | 내과, 외과, 정형외과, 소아과 등 30개 과목 → 코드 자동 변환 |
| **증상 기반 추천** | 에이전트가 사용자 증상을 분석하여 적절한 진료과목을 자동 추론 (예: 발열/구토 → 내과) |

### Tool 3: `get_drug_info` - 의약품 정보 조회

| 항목 | 내용 |
|------|------|
| **데이터 출처** | 식품의약품안전처 의약품개요정보(e약은요) API |
| **입력** | `drug_name`: 의약품명 (예: `"타이레놀"`, `"아스피린"`) |
| **출력** | 제조사, 효능, 용법, 주의사항, 부작용 (최대 3건) |

### Tool 4: `search_emergency_rooms` - 응급실 실시간 정보

| 항목 | 내용 |
|------|------|
| **데이터 출처** | 국립중앙의료원 응급의료정보제공서비스 API |
| **입력** | `region`: 지역명 (예: `"서울"`, `"서울 강남구"`) |
| **출력** | 병원명, 주소, 응급실 전화, 가용 병상 수, 수술실 가용 여부 (최대 5건) |
| **검색 범위** | 시도 → 시군구까지 (읍면동 미지원) |
| **응답 형식** | XML (내부에서 `xml.etree.ElementTree`로 파싱) |
| **지역 매핑** | `"서울"` → `"서울특별시"` 등 17개 시도 자동 변환 |

### Tool 5: `search_pharmacies` - 약국 검색

| 항목 | 내용 |
|------|------|
| **데이터 출처** | 건강보험심사평가원 약국정보서비스 API |
| **입력** | `region`: 지역명 (예: `"강남구"`, `"종로구"`, `"서울 중랑구 중화동"`) |
| **출력** | 약국명, 주소, 전화번호 (최대 5건) |
| **검색 범위** | 시도 → 시군구 → 읍면동 |

### 지역 검색 입력 형식

`parse_region()` 함수가 사용자의 지역명 입력을 자동으로 파싱하여 각 API에 맞는 파라미터로 변환합니다.

| 입력 형태 | 예시 | 병원/약국 | 응급실 |
|---|---|---|---|
| 시도 | `"서울"`, `"부산"` | O | O |
| 시군구 | `"강남구"`, `"해운대구"` | O | O |
| 시도 + 시군구 | `"서울 중랑구"`, `"부산 동구"` | O | O |
| 시군구 + 읍면동 | `"강남구 역삼동"` | O | O (읍면동 무시) |
| 시도 + 시군구 + 읍면동 | `"서울 중랑구 중화동"` | O | O (읍면동 무시) |
| 읍면동만 단독 | `"중화동"`, `"역삼동"` | **X** | **X** |

- **읍면동만 단독 입력은 지원하지 않습니다.** 같은 동 이름이 여러 시군구에 존재할 수 있으므로 (예: 신사동 → 강남구/은평구) 시군구 이상의 정보가 필요합니다.
- 중복 시군구명(중구, 동구, 서구, 남구, 북구, 강서구)은 반드시 시도와 함께 입력해야 합니다. (예: `"서울 중구"`, `"부산 동구"`)
- 응급실 API는 읍면동 파라미터를 지원하지 않으므로, 읍면동이 포함되어도 시군구 레벨로 검색됩니다.

## 환경 준비 및 설치

### 1. 사전 요구사항

- Python 3.11 이상 3.13 이하
- `uv` 패키지 매니저:
  ```bash
  # macOS / Linux / Windows (WSL)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. 의존성 설치

```bash
cd agent
uv sync
```

실행 후 프로젝트 디렉토리에 `.venv` 폴더가 생성됩니다.

### 3. 환경 변수 설정

```bash
cp env.sample .env
```

`.env` 파일을 열고 아래 항목을 설정합니다:

```env
# API 라우트 prefix
API_V1_PREFIX="/api/v1"

# CORS 허용 Origin (React UI 주소)
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# =====================================================
# OpenAI 설정
# =====================================================
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1

# =====================================================
# Elasticsearch 설정
# =====================================================
ES_URL=https://your-elasticsearch-url
ES_USERNAME=elastic
ES_PASSWORD=your_password
ES_INDEX=edu-collection

# =====================================================
# 공공데이터포털 API 키
# https://www.data.go.kr 에서 아래 4개 서비스 활용 신청 후 발급
# - 건강보험심사평가원_병원정보서비스
# - 식품의약품안전처_의약품개요정보(e약은요)
# - 국립중앙의료원_응급의료정보제공서비스
# - 건강보험심사평가원_약국정보서비스
# =====================================================
PUBLIC_DATA_API_KEY=your_api_key
```

### 4. 공공데이터포털 API 키 발급 방법

1. [공공데이터포털](https://www.data.go.kr) 회원가입 및 로그인
2. 아래 4개 API 서비스를 각각 검색하여 **활용 신청**
   - `건강보험심사평가원_병원정보서비스` (병원 검색)
   - `식품의약품안전처_의약품개요정보(e약은요)` (의약품 조회)
   - `국립중앙의료원_응급의료정보제공서비스` (응급실 실시간 정보)
   - `건강보험심사평가원_약국정보서비스` (약국 검색)
3. 승인 후 마이페이지에서 **일반 인증키 (Encoding)** 를 복사
4. `.env` 파일의 `PUBLIC_DATA_API_KEY`에 설정 (4개 서비스 모두 동일 키 사용)

### 5. 서버 실행

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API 문서 (Swagger UI): http://localhost:8000/docs
- 헬스 체크: http://localhost:8000/health

### 6. UI 연동 (선택)

별도의 React UI 프로젝트(`ui/`)와 함께 사용합니다.

```bash
cd ui
npm install
npm run dev    # http://localhost:5173
```

## 프로젝트 구조

```
agent/
├── app/
│   ├── main.py                         # FastAPI 앱 진입점, CORS, 미들웨어 설정
│   ├── agents/
│   │   ├── medical_agent.py            # 에이전트 팩토리 (create_agent)
│   │   ├── prompts.py                  # 시스템 프롬프트 (도구 설명, 답변 규칙)
│   │   ├── region_codes.py             # 전국 시도/시군구 코드 매핑 + parse_region()
│   │   └── tools.py                    # 5개 도구 정의 (async) + ES Retriever + 유틸리티
│   ├── api/
│   │   └── routes/
│   │       ├── chat.py                 # POST /api/v1/chat (SSE 스트리밍 응답)
│   │       └── threads.py              # GET /api/v1/threads (대화 목록/상세 조회)
│   ├── core/
│   │   └── config.py                   # 환경 설정 (pydantic-settings, .env 로드)
│   ├── models/
│   │   ├── __init__.py                 # LangChainMessage, 응답 DTO 정의
│   │   ├── chat.py                     # ChatRequest / ChatResponse 모델
│   │   └── threads.py                  # 대화 스레드 모델
│   ├── services/
│   │   ├── agent_service.py            # 에이전트 실행 + SSE 스트리밍 로직
│   │   ├── conversation_service.py     # 대화 세션 관리 (메모리 기반)
│   │   └── threads_service.py          # 대화 목록 JSON 조회
│   └── utils/
│       ├── logger.py                   # 커스텀 로거 + @log_execution 데코레이터
│       └── read_json.py                # JSON 파일 읽기 유틸리티
├── tests/                              # pytest 테스트 (실제 API 호출 통합 테스트)
├── env.sample                          # 환경 변수 샘플
├── pyproject.toml                      # 프로젝트 설정 및 의존성 (uv)
└── README.md
```

### 주요 모듈 설명

| 모듈 | 역할 |
|------|------|
| `agents/medical_agent.py` | `create_agent()`로 LangGraph 에이전트를 생성. LLM + 도구 목록 + 시스템 프롬프트 + 체크포인터를 조합 |
| `agents/region_codes.py` | 전국 17개 시도 + ~250개 시군구 코드 매핑. `parse_region()`으로 지역명 → 숫자 코드(병원/약국) + 한글명(응급실) 동시 반환 |
| `agents/tools.py` | 5개 `@tool` 데코레이터 함수 정의 (4개 async). ES Retriever, 공공데이터 API 호출, XML/JSON 파싱, 진료과목 코드 매핑 |
| `agents/prompts.py` | 의료 AI 페르소나, 도구 사용 가이드, 답변 규칙, 의료 면책 조항을 포함한 시스템 프롬프트 |
| `services/agent_service.py` | `MemorySaver` 기반 체크포인터로 대화 기록 유지. `astream(stream_mode="updates")`으로 에이전트 실행 결과를 SSE 이벤트로 변환 |
| `api/routes/chat.py` | SSE `StreamingResponse` 생성. 초기 "Planning" 이벤트 전송 후 에이전트 스트림 연결 |

## API 엔드포인트

| Method | Path | 설명 | 요청 Body |
|--------|------|------|-----------|
| `GET` | `/` | API 정보 | - |
| `GET` | `/health` | 헬스 체크 | - |
| `POST` | `/api/v1/chat` | 채팅 (SSE 스트리밍) | `{ "thread_id": "uuid", "message": "질문" }` |
| `GET` | `/api/v1/favorites/questions` | 즐겨찾기 질문 목록 | - |
| `GET` | `/api/v1/threads` | 대화 목록 조회 | - |
| `GET` | `/api/v1/threads/{thread_id}` | 대화 상세 조회 | - |

### 채팅 API 사용 예시

```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6", "message": "결핵 치료 방법 알려줘"}'
```

## 대화 메모리

`MemorySaver` (인메모리) 기반 체크포인터를 사용하여 동일한 `thread_id`의 대화 문맥을 유지합니다.

- 같은 `thread_id`로 요청하면 이전 대화를 기억하고 맥락에 맞는 답변을 생성합니다.
- 서버 재시작 시 대화 기록이 초기화됩니다.
- 향후 `SqliteSaver` 또는 `PostgresSaver`로 교체하여 영구 저장이 가능합니다.

## 질문 예시

### 의료 문서 검색
- "결핵 치료 방법 알려줘"
- "천식 증상이 뭐야?"
- "고혈압 관리법 알려줘"

### 병원 검색
- "강남구 내과 병원 추천해줘"
- "종로구 정형외과 찾아줘"
- "부산 소아과 병원 알려줘"
- "구토와 고열이 있는데 서울 중랑구 중화동 근처 병원 알려줘" (증상 기반 자동 진료과목 추론)

### 의약품 정보
- "타이레놀 부작용 알려줘"
- "아스피린 복용법 알려줘"
- "아목시실린 효능이 뭐야?"

### 응급실 실시간 정보
- "서울 응급실 빈 병상 알려줘"
- "부산 응급실 현황 알려줘"

### 약국 검색
- "종로구 약국 찾아줘"
- "강남구 약국 알려줘"

### 복합 질문
- "두통이 심한데 타이레놀 먹어도 돼? 근처 강남구 내과도 알려줘"
- "서울 응급실 빈 병상이랑 종로구 약국도 같이 찾아줘"

## 주의사항

- 모든 답변은 **일반적인 의료 정보 제공**을 목적으로 하며, 실제 진료나 처방을 대체하지 않습니다.
- 정확한 판단을 위해 반드시 **전문의를 방문**하세요.
- 공공데이터포털 API는 일일 호출 제한이 있을 수 있으며, API 키 미설정 시 해당 도구는 안내 메시지를 반환합니다.
