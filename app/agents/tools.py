import xml.etree.ElementTree as ET
from typing import Optional

import httpx
from langchain_core.tools import tool

from app.agents.region_codes import parse_region
from app.core.config import settings


# ============================================================
# Tool 1: 병원 정보 검색 (건강보험심사평가원 API)
# ============================================================

@tool
async def search_hospitals(region: str, specialty: Optional[str] = None) -> str:
    """지역과 진료과목을 기반으로 병원/의원 정보를 검색합니다.

    Args:
        region: 검색할 지역명 (예: '서울', '강남구', '부산 중구')
        specialty: 진료과목 (예: '내과', '정형외과', '소아과'). 선택사항.
    """
    parsed = parse_region(region)
    url = settings.HOSPITAL_API_URL
    params = {
        "serviceKey": settings.PUBLIC_DATA_API_KEY,
        "numOfRows": "5",
        "pageNo": "1",
        "sidoCd": parsed["sidoCd"],
        "sgguCd": parsed["sgguCd"],
        "emdongNm": parsed.get("emdongNm", "") if parsed["sidoCd"] else parsed["raw"],
        "yadmNm": "",
        "zipCd": "",
        "_type": "json",
    }
    if specialty:
        params["dgsbjtCd"] = _get_specialty_code(specialty)

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
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


# ============================================================
# Tool 3: 의약품 정보 조회 (의약품안전나라 API)
# ============================================================

@tool
async def get_drug_info(drug_name: str) -> str:
    """의약품명으로 효능, 용법, 부작용 등 상세 정보를 조회합니다.

    Args:
        drug_name: 검색할 의약품명 (예: '타이레놀', '아스피린', '아목시실린')
    """
    url = settings.DRUG_API_URL
    params = {
        "serviceKey": settings.PUBLIC_DATA_API_KEY,
        "itemName": drug_name,
        "numOfRows": "3",
        "pageNo": "1",
        "type": "json",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
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


# ============================================================
# Tool 4: 응급실 실시간 정보 (국립중앙의료원 API)
# ============================================================

@tool
async def search_emergency_rooms(region: str) -> str:
    """지역 기반으로 응급실의 실시간 병상 가용 정보를 조회합니다.

    Args:
        region: 검색할 지역명 (예: '서울', '서울 강남구', '부산')
    """
    parsed = parse_region(region)
    url = settings.EMERGENCY_API_URL
    params = {
        "serviceKey": settings.PUBLIC_DATA_API_KEY,
        "STAGE1": parsed["sido_name"] if parsed["sido_name"] else parsed["raw"],
        "STAGE2": parsed["sggu_name"],
        "pageNo": "1",
        "numOfRows": "5",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        response_text = response.text

    root = ET.fromstring(response_text)

    items = root.findall(".//item")
    if not items:
        return f"'{region}' 지역의 응급실 정보를 찾을 수 없습니다."

    results = []
    for i, item in enumerate(items[:5], 1):
        name = _xml_text(item, "dutyName")
        addr = _xml_text(item, "dutyAddr")
        tel = _xml_text(item, "dutyTel3")
        hvec = _xml_text(item, "hvec")
        hvoc = _xml_text(item, "hvoc")

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


# ============================================================
# Tool 5: 약국 검색 (건강보험심사평가원 API)
# ============================================================

@tool
async def search_pharmacies(region: str) -> str:
    """지역 기반으로 약국 정보를 검색합니다.

    Args:
        region: 검색할 지역명 (예: '강남구', '종로구', '서울 중구')
    """
    parsed = parse_region(region)
    url = settings.PHARMACY_API_URL
    params = {
        "serviceKey": settings.PUBLIC_DATA_API_KEY,
        "numOfRows": "5",
        "pageNo": "1",
        "sidoCd": parsed["sidoCd"],
        "sgguCd": parsed["sgguCd"],
        "emdongNm": parsed.get("emdongNm", "") if parsed["sidoCd"] else parsed["raw"],
        "_type": "json",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
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


# ============================================================
# 유틸리티 함수
# ============================================================

def _xml_text(item, tag: str) -> str:
    """XML 엘리먼트에서 텍스트 추출"""
    el = item.find(tag)
    return el.text if el is not None and el.text else "정보없음"


# ============================================================
# 진료과목 코드 매핑 (건강보험심사평가원 기준)
# ============================================================

def _get_specialty_code(specialty: str) -> str:
    """진료과목명 → API 코드 변환 (예: "정형외과" → "05")"""
    codes = {
        "내과": "01", "신경과": "02", "정신건강의학과": "03", "외과": "04",
        "정형외과": "05", "신경외과": "06", "흉부외과": "07", "성형외과": "08",
        "마취통증의학과": "09", "산부인과": "10", "소아청소년과": "11", "소아과": "11",
        "안과": "12", "이비인후과": "13", "피부과": "14", "비뇨의학과": "15",
        "영상의학과": "16", "방사선종양학과": "17", "병리과": "18", "진단검사의학과": "19",
        "직업환경의학과": "20", "재활의학과": "21", "핵의학과": "22",
        "가정의학과": "23", "응급의학과": "24",
        "치과": "49", "한방내과": "80", "한방부인과": "81", "한방소아과": "82",
        "한방안이비인후피부과": "83", "한방신경정신과": "84", "침구과": "85",
        "한방재활의학과": "86", "사상체질과": "87", "한방응급": "88",
    }
    return codes.get(specialty, "")
