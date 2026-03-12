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
# Tool 4: 응급실 실시간 정보 (국립중앙의료원 API)
# ============================================================

@tool
def search_emergency_rooms(region: str) -> str:
    """지역 기반으로 응급실의 실시간 병상 가용 정보를 조회합니다.

    Args:
        region: 검색할 지역명 (예: '서울', '강남구', '부산', '대구')
    """
    api_key = settings.PUBLIC_DATA_API_KEY
    if not api_key:
        return (
            "응급실 정보 조회 기능을 사용하려면 공공데이터포털(data.go.kr)에서 "
            "'국립중앙의료원_응급의료정보제공서비스' API 키를 발급받아 "
            ".env 파일의 PUBLIC_DATA_API_KEY에 설정해주세요."
        )

    try:
        # 지역명 → 시도 코드 변환
        sido_code = _get_sido_code(region)

        url = "http://apis.data.go.kr/B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire"
        params = {
            "serviceKey": api_key,
            "STAGE1": sido_code if sido_code else region,
            "STAGE2": "" if sido_code else "",
            "pageNo": "1",
            "numOfRows": "5",
        }

        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params)

        # XML 응답 파싱
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.text)

        items = root.findall(".//item")
        if not items:
            return f"'{region}' 지역의 응급실 정보를 찾을 수 없습니다."

        results = []
        for i, item in enumerate(items[:5], 1):
            name = _xml_text(item, "dutyName")
            addr = _xml_text(item, "dutyAddr")
            tel = _xml_text(item, "dutyTel3")
            hvec = _xml_text(item, "hvec")  # 응급실 일반 병상 수
            hvoc = _xml_text(item, "hvoc")  # 수술실 가용 여부
            hvs01 = _xml_text(item, "hvs01")  # 일반 입원실

            # 수술실 가용 여부 판단 (숫자가 아닌 경우 '불가' 처리)
            try:
                surgery_available = "가능" if hvoc and int(hvoc) > 0 else "불가"
            except (ValueError, TypeError):
                surgery_available = "정보없음"

            results.append(
                f"[{i}] {name}\n"
                f"   주소: {addr}\n"
                f"   응급실 전화: {tel}\n"
                f"   응급실 가용 병상: {hvec}개\n"
                f"   수술실 가용: {surgery_available}"
            )

        return "\n\n".join(results)
    except Exception as e:
        custom_logger.error(f"응급실 정보 조회 오류: {e}")
        return f"응급실 정보 조회 중 오류가 발생했습니다: {str(e)}"


# ============================================================
# Tool 5: 약국 검색 (건강보험심사평가원 API)
# ============================================================

@tool
def search_pharmacies(region: str) -> str:
    """지역 기반으로 약국 정보를 검색합니다.

    Args:
        region: 검색할 지역명 (예: '강남구', '종로구', '해운대구')
    """
    api_key = settings.PUBLIC_DATA_API_KEY
    if not api_key:
        return (
            "약국 검색 기능을 사용하려면 공공데이터포털(data.go.kr)에서 "
            "'건강보험심사평가원_약국정보서비스' API 키를 발급받아 "
            ".env 파일의 PUBLIC_DATA_API_KEY에 설정해주세요."
        )

    try:
        url = "http://apis.data.go.kr/B551182/pharmacyInfoService/getParmacyBasisList"
        params = {
            "serviceKey": api_key,
            "numOfRows": "5",
            "pageNo": "1",
            "emdongNm": region,
            "_type": "json",
        }

        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params)
            data = response.json()

        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
        if not items:
            return f"'{region}' 지역에서 약국을 찾을 수 없습니다."

        if isinstance(items, dict):
            items = [items]

        results = []
        for i, item in enumerate(items[:5], 1):
            name = item.get("yadmNm", "정보없음")
            addr = item.get("addr", "정보없음")
            tel = item.get("telno", "정보없음")
            results.append(
                f"[{i}] {name}\n"
                f"   주소: {addr}\n"
                f"   전화: {tel}"
            )

        return "\n\n".join(results)
    except Exception as e:
        custom_logger.error(f"약국 검색 오류: {e}")
        return f"약국 검색 중 오류가 발생했습니다: {str(e)}"


# ============================================================
# 유틸리티 함수
# ============================================================

def _xml_text(item, tag: str) -> str:
    """XML 엘리먼트에서 텍스트 추출"""
    el = item.find(tag)
    return el.text if el is not None and el.text else "정보없음"


def _get_sido_code(region: str) -> str:
    """지역명을 시도명으로 매핑"""
    mapping = {
        "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시",
        "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
        "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도",
        "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
        "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도",
        "경남": "경상남도", "제주": "제주특별자치도",
    }
    return mapping.get(region, "")


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
