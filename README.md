# Medical AI Agent

LangChain + LangGraph 기반의 의료 전문 AI 에이전트입니다. Elasticsearch BM25 검색과 공공데이터포털 API를 활용하여 의료 정보, 병원, 의약품, 응급실, 약국 정보를 제공합니다.

## 기술 스택

- **Backend**: FastAPI, LangChain v1.0, LangGraph
- **LLM**: OpenAI GPT-4.1
- **검색엔진**: Elasticsearch (BM25 기반 의료 문서 검색)
- **외부 API**: 공공데이터포털 (병원, 의약품, 응급실, 약국)
- **패키지 관리**: uv

## 에이전트 도구 (Tools)

| 도구 | 설명 | 데이터 출처 |
|------|------|-------------|
| `search_medical_info` | 증상, 질병, 치료법 등 의료 문서 검색 | Elasticsearch (BM25) |
| `search_hospitals` | 지역/진료과목 기반 병원 검색 | 건강보험심사평가원 API |
| `get_drug_info` | 의약품 효능, 용법, 부작용 조회 | 식품의약품안전처 API |
| `search_emergency_rooms` | 응급실 실시간 병상 가용 정보 | 국립중앙의료원 API |
| `search_pharmacies` | 지역 기반 약국 검색 | 건강보험심사평가원 API |

## 환경 준비 및 설치

### 1. 사전 요구사항
* Python 3.11 이상 3.13 이하
* `uv` 패키지 매니저:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. 의존성 설치
```bash
uv sync
```

### 3. 환경 변수 설정
```bash
cp env.sample .env
```

`.env` 파일에 아래 항목을 설정합니다:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1

# Elasticsearch
ES_URL=https://your-es-url
ES_USERNAME=elastic
ES_PASSWORD=your_password
ES_INDEX=your_index

# 공공데이터포털 API 키 (https://www.data.go.kr)
PUBLIC_DATA_API_KEY=your_api_key
```

### 4. 서버 실행
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API 문서: `http://localhost:8000/docs`

## 프로젝트 구조

```
agent/
├── app/
│   ├── api/
│   │   └── routes/
│   │       └── chat.py            # POST /api/v1/chat (SSE 스트리밍)
│   ├── agents/
│   │   ├── medical_agent.py       # ReAct 에이전트 생성
│   │   ├── prompts.py             # 시스템 프롬프트
│   │   └── tools.py               # 5개 도구 정의
│   ├── core/
│   │   └── config.py              # 환경 설정 (pydantic-settings)
│   ├── models/                    # 요청/응답 모델
│   ├── services/
│   │   └── agent_service.py       # 에이전트 실행 및 SSE 스트리밍
│   ├── utils/
│   └── main.py                    # FastAPI 앱 진입점
├── tests/
├── pyproject.toml
└── README.md
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/` | API 정보 |
| `GET` | `/health` | 헬스 체크 |
| `POST` | `/api/v1/chat` | 채팅 (SSE 스트리밍 응답) |

## 질문 예시

```
- "결핵 치료 방법 알려줘"
- "강남구 내과 병원 추천해줘"
- "타이레놀 부작용 알려줘"
- "서울 응급실 빈 병상 알려줘"
- "종로구 약국 찾아줘"
```
