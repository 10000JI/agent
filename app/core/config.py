from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class OpikSettings(BaseSettings):
    """Opik 트레이싱 설정 (.env에서 OPIK__ 접두사로 로드)"""

    URL_OVERRIDE: str | None = Field(default=None, description="Opik base URL")
    API_KEY: str | None = Field(default=None, description="Opik Cloud API key")
    WORKSPACE: str | None = Field(default=None, description="Opik workspace name")
    PROJECT: str | None = Field(default=None, description="Opik project name")


class Settings(BaseSettings):
    # ── FastAPI 서버 설정 ──
    API_V1_PREFIX: str
    CORS_ORIGINS: List[str] = ["*"]

    # ── LLM 설정 (ReAct 에이전트의 두뇌) ──
    OPENAI_API_KEY: str
    OPENAI_MODEL: str  # 예: gpt-4.1

    # ── Elasticsearch 설정 (의료 문서 BM25 검색) ──
    ES_URL: str
    ES_USERNAME: str
    ES_PASSWORD: str
    ES_INDEX: str  # 의료 문서가 저장된 인덱스명

    # ── 공공데이터포털 API (data.go.kr) ──
    # 5개 도구 중 4개(병원, 약물, 응급실, 약국)가 이 API를 사용합니다.
    PUBLIC_DATA_API_KEY: str

    # 각 공공 API 엔드포인트
    HOSPITAL_API_URL: str = "https://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList"          # 건강보험심사평가원
    DRUG_API_URL: str = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"       # 식품의약품안전처
    EMERGENCY_API_URL: str = "https://apis.data.go.kr/B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire"  # 국립중앙의료원
    PHARMACY_API_URL: str = "https://apis.data.go.kr/B551182/pharmacyInfoService/getParmacyBasisList"      # 건강보험심사평가원

    # ── Cohere ReRank 설정 (선택) ──
    COHERE_API_KEY: str | None = None

    # ── 에이전트 동작 설정 ──
    # ReAct 루프의 최대 반복 횟수 (도구 호출 → 판단 → 도구 호출 ... 무한 루프 방지)
    DEEPAGENT_RECURSION_LIMIT: int = 20

    # ── Opik 관찰성 설정 (선택) ──
    # .env에 OPIK__URL_OVERRIDE 등을 설정하면 자동으로 로드됩니다.
    OPIK: OpikSettings | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # OPIK__PROJECT → OpikSettings.PROJECT로 매핑
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

