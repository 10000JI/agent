from langchain_core.tools import tool
from typing import Optional
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchRetriever
from app.core.config import settings
from app.utils.logger import custom_logger
import httpx


# ============================================================
# Elasticsearch 클라이언트 및 Retriever 설정
# ============================================================

def _get_es_client() -> Elasticsearch:
    """Elasticsearch 클라이언트 생성"""
    return Elasticsearch(
        settings.ES_URL,
        basic_auth=(settings.ES_USERNAME, settings.ES_PASSWORD),
        verify_certs=True,
    )


def _bm25_query(search_query: str) -> dict:
    """BM25 검색 쿼리 생성"""
    return {
        "query": {
            "match": {
                "content": {
                    "query": search_query,
                    "analyzer": "standard"
                }
            }
        },
        "size": 5,
        "_source": ["c_id", "content", "source_spec", "creation_year"]
    }


def get_medical_retriever() -> ElasticsearchRetriever:
    """BM25 기반 ElasticsearchRetriever 생성 (ES 클라이언트 주입)"""
    es_client = _get_es_client()
    return ElasticsearchRetriever(
        index_name=settings.ES_INDEX,
        body_func=_bm25_query,
        content_field="content",
        client=es_client,
    )


# ============================================================
# Tool 1: 의료 문서 검색 (Elasticsearch BM25)
# ============================================================

@tool
def search_medical_info(query: str) -> str:
    """주어진 증상, 질병명, 치료법 등을 기반으로 Elasticsearch에서 관련 의료 정보를 검색합니다.

    Args:
        query: 검색할 증상, 질병명, 치료법 등 (예: '결핵 치료', '천식 증상', '응급처치')
    """
    try:
        retriever = get_medical_retriever()
        docs = retriever.invoke(query)

        if not docs:
            return "관련 의료 정보를 찾을 수 없습니다."

        results = []
        for i, doc in enumerate(docs[:5], 1):
            content = doc.page_content[:500]
            # metadata가 _source 안에 중첩될 수 있음
            meta = doc.metadata.get("_source", doc.metadata)
            source = meta.get("source_spec", "unknown")
            year = meta.get("creation_year", "unknown")
            results.append(f"[문서 {i}] (출처: {source}, 연도: {year})\n{content}")

        return "\n\n---\n\n".join(results)
    except Exception as e:
        custom_logger.error(f"의료 문서 검색 오류: {e}")
        return f"검색 중 오류가 발생했습니다: {str(e)}"


# ============================================================
# Tool 2: 병원 정보 검색 (건강보험심사평가원 API)
# ============================================================

@tool
def search_hospitals(region: str, specialty: Optional[str] = None) -> str:
    """지역과 진료과목을 기반으로 병원/의원 정보를 검색합니다.

    Args:
        region: 검색할 지역명 (예: '서울', '강남구', '부산')
        specialty: 진료과목 (예: '내과', '정형외과', '소아과'). 선택사항.
    """
    api_key = settings.PUBLIC_DATA_API_KEY
    if not api_key:
        return (
            "병원 검색 기능을 사용하려면 공공데이터포털(data.go.kr)에서 "
            "'건강보험심사평가원_병원정보서비스' API 키를 발급받아 "
            ".env 파일의 PUBLIC_DATA_API_KEY에 설정해주세요."
        )

    try:
        url = "http://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList"
        params = {
            "serviceKey": api_key,
            "numOfRows": "5",
            "pageNo": "1",
            "sidoCd": "",
            "sgguCd": "",
            "emdongNm": region,
            "yadmNm": "",
            "zipCd": "",
            "_type": "json",
        }
        if specialty:
            params["dgsbjtCd"] = _get_specialty_code(specialty)

        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params)
            data = response.json()

        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        if not items:
            return f"'{region}' 지역에서 관련 병원을 찾을 수 없습니다."

        if isinstance(items, dict):
            items = [items]

        results = []
        for i, item in enumerate(items[:5], 1):
            name = item.get("yadmNm", "정보없음")
            addr = item.get("addr", "정보없음")
            tel = item.get("telno", "정보없음")
            category = item.get("clCdNm", "정보없음")
            results.append(
                f"[{i}] {name}\n"
                f"   종류: {category}\n"
                f"   주소: {addr}\n"
                f"   전화: {tel}"
            )

        return "\n\n".join(results)
    except Exception as e:
        custom_logger.error(f"병원 검색 오류: {e}")
        return f"병원 검색 중 오류가 발생했습니다: {str(e)}"


# ============================================================
# Tool 3: 의약품 정보 조회 (의약품안전나라 API)
# ============================================================

@tool
def get_drug_info(drug_name: str) -> str:
    """의약품명으로 효능, 용법, 부작용 등 상세 정보를 조회합니다.

    Args:
        drug_name: 검색할 의약품명 (예: '타이레놀', '아스피린', '아목시실린')
    """
    api_key = settings.PUBLIC_DATA_API_KEY
    if not api_key:
        return (
            "의약품 정보 조회 기능을 사용하려면 공공데이터포털(data.go.kr)에서 "
            "'식품의약품안전처_의약품개요정보' API 키를 발급받아 "
            ".env 파일의 PUBLIC_DATA_API_KEY에 설정해주세요."
        )

    try:
        url = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
        params = {
            "serviceKey": api_key,
            "itemName": drug_name,
            "numOfRows": "3",
            "pageNo": "1",
            "type": "json",
        }

        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params)
            data = response.json()

        items = data.get("body", {}).get("items", [])
        if not items:
            return f"'{drug_name}'에 대한 의약품 정보를 찾을 수 없습니다."

        results = []
        for i, item in enumerate(items[:3], 1):
            name = item.get("itemName", "정보없음")
            company = item.get("entpName", "정보없음")
            effect = item.get("efcyQesitm", "정보없음")
            usage = item.get("useMethodQesitm", "정보없음")
            warning = item.get("atpnQesitm", "정보없음")
            side_effect = item.get("seQesitm", "정보없음")

            results.append(
                f"[{i}] {name} ({company})\n"
                f"   효능: {effect[:200]}\n"
                f"   용법: {usage[:200]}\n"
                f"   주의사항: {warning[:200]}\n"
                f"   부작용: {side_effect[:200]}"
            )

        return "\n\n".join(results)
    except Exception as e:
        custom_logger.error(f"의약품 정보 조회 오류: {e}")
        return f"의약품 정보 조회 중 오류가 발생했습니다: {str(e)}"


# ============================================================
# 진료과목 코드 매핑 (건강보험심사평가원 기준)
# ============================================================

def _get_specialty_code(specialty: str) -> str:
    """진료과목명을 코드로 변환"""
    codes = {
        "내과": "01", "신경과": "02", "정신건강의학과": "03", "외과": "04",
        "정형외과": "05", "신경외과": "06", "흉부외과": "07", "성형외과": "08",
        "마취통증의학과": "09", "산부인과": "10", "소아청소년과": "11", "소아과": "11",
        "안과": "12", "이비인후과": "13", "피부과": "14", "비뇨의학과": "15",
        "영상의학과": "16", "방사선종양학과": "17", "병리과": "18", "진단검사의학과": "19",
        "재활의학과": "20", "핵의학과": "21", "가정의학과": "22", "응급의학과": "23",
        "치과": "49", "한방내과": "80", "한방부인과": "81", "한방소아과": "82",
        "한방안이비인후피부과": "83", "한방신경정신과": "84", "침구과": "85",
        "한방재활의학과": "86", "사상체질과": "87", "한방응급": "88",
    }
    return codes.get(specialty, "")
