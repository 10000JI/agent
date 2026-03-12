"""5개 도구 실제 API 호출 테스트.

모든 테스트는 실제 외부 API(공공데이터포털, Elasticsearch)를 호출한다.
pytest-asyncio 사용.
"""
import pytest
from app.agents.tools import (
    search_medical_info,
    search_hospitals,
    get_drug_info,
    search_emergency_rooms,
    search_pharmacies,
)


class TestSearchMedicalInfo:
    """Tool 1: search_medical_info (Elasticsearch BM25)"""

    def test_search_returns_documents(self):
        """'결핵 치료' 검색 → 문서 결과 반환"""
        result = search_medical_info.invoke({"query": "결핵 치료"})
        assert "[문서 1]" in result
        assert len(result) > 0


class TestSearchHospitals:
    """Tool 2: search_hospitals (건강보험심사평가원 API)"""

    @pytest.mark.asyncio
    async def test_unique_sggu(self):
        """고유 시군구 '강남구' → 병원 결과 반환"""
        result = await search_hospitals.ainvoke({"region": "강남구"})
        assert "찾을 수 없습니다" not in result
        assert "[1]" in result

    @pytest.mark.asyncio
    async def test_sido_level(self):
        """시도 레벨 '서울' → 병원 결과 반환"""
        result = await search_hospitals.ainvoke({"region": "서울"})
        assert "찾을 수 없습니다" not in result

    @pytest.mark.asyncio
    async def test_with_specialty(self):
        """진료과목 지정 '강남구, 내과' → 결과 반환"""
        result = await search_hospitals.ainvoke({"region": "강남구", "specialty": "내과"})
        assert "찾을 수 없습니다" not in result

    @pytest.mark.asyncio
    async def test_emdong_level(self):
        """읍면동 '서울 중랑구 중화동' → 해당 동 병원 반환"""
        result = await search_hospitals.ainvoke({"region": "서울 중랑구 중화동"})
        assert "찾을 수 없습니다" not in result
        assert "중화동" in result


class TestGetDrugInfo:
    """Tool 3: get_drug_info (의약품안전나라 API)"""

    @pytest.mark.asyncio
    async def test_drug_info(self):
        """'타이레놀' → 효능, 용법 포함"""
        result = await get_drug_info.ainvoke({"drug_name": "타이레놀"})
        assert "효능" in result
        assert "용법" in result


class TestSearchEmergencyRooms:
    """Tool 4: search_emergency_rooms (국립중앙의료원 API)"""

    @pytest.mark.asyncio
    async def test_sido_level(self):
        """시도 '서울' → 응급실 가용 병상 포함"""
        result = await search_emergency_rooms.ainvoke({"region": "서울"})
        assert "응급실 가용 병상" in result

    @pytest.mark.asyncio
    async def test_sggu_level(self):
        """시군구 '서울 강남구' → STAGE2 세분화 동작"""
        result = await search_emergency_rooms.ainvoke({"region": "서울 강남구"})
        assert "응급실" in result or "찾을 수 없습니다" in result


class TestSearchPharmacies:
    """Tool 5: search_pharmacies (건강보험심사평가원 API)"""

    @pytest.mark.asyncio
    async def test_unique_sggu(self):
        """고유 시군구 '종로구' → 약국 결과 반환"""
        result = await search_pharmacies.ainvoke({"region": "종로구"})
        assert "찾을 수 없습니다" not in result
        assert "[1]" in result

    @pytest.mark.asyncio
    async def test_sido_level(self):
        """시도 '서울' → 약국 결과 반환"""
        result = await search_pharmacies.ainvoke({"region": "서울"})
        assert "찾을 수 없습니다" not in result

    @pytest.mark.asyncio
    async def test_emdong_level(self):
        """읍면동 '서울 중랑구 중화동' → 해당 동 약국 반환"""
        result = await search_pharmacies.ainvoke({"region": "서울 중랑구 중화동"})
        assert "찾을 수 없습니다" not in result
        assert "중화동" in result
