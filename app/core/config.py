from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class OpikSettings(BaseSettings):
    """Opik configuration."""

    URL_OVERRIDE: str | None = Field(default=None, description="Opik base URL")
    # Optional if you are using Opik Cloud:
    API_KEY: str | None = Field(default=None, description="opik cloud api key here")
    WORKSPACE: str | None = Field(default=None, description="your workspace name")
    PROJECT: str | None = Field(default=None, description="your project name")


class Settings(BaseSettings):
    # API 설정
    API_V1_PREFIX: str

    CORS_ORIGINS: List[str] = ["*"]
    
    # IMP: LangChain 객체 및 LLM 연동에 사용되는 필수 설정값(API Key 등)
    # LangChain 설정
    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    
    # Elasticsearch 설정
    ES_URL: str
    ES_USERNAME: str
    ES_PASSWORD: str
    ES_INDEX: str

    # 공공데이터포털 API 키 (병원 검색, 의약품 정보)
    PUBLIC_DATA_API_KEY: str

    # 공공데이터포털 API URL
    HOSPITAL_API_URL: str = "http://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList"
    DRUG_API_URL: str = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
    EMERGENCY_API_URL: str = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire"
    PHARMACY_API_URL: str = "http://apis.data.go.kr/B551182/pharmacyInfoService/getParmacyBasisList"

    # DeepAgents 설정
    DEEPAGENT_RECURSION_LIMIT: int = 20

    OPIK: OpikSettings | None = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=True,
        extra="ignore",
    )

settings = Settings()

