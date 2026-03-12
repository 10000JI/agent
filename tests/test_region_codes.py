"""parse_region() 순수 함수 테스트.

외부 의존성 없이 지역명 매핑 정확성을 검증한다.
"""
import pytest
from app.agents.region_codes import parse_region


class TestParseRegion:
    """parse_region() 매핑 정확성 테스트"""

    def test_sido_level(self):
        """시도 레벨 입력 → sidoCd, sido_name 매핑"""
        result = parse_region("서울")
        assert result["sidoCd"] == "110000"
        assert result["sido_name"] == "서울특별시"
        assert result["sgguCd"] == ""

    def test_unique_sggu(self):
        """고유 시군구 → sidoCd + sgguCd 매핑"""
        result = parse_region("강남구")
        assert result["sidoCd"] == "110000"
        assert result["sgguCd"] == "110001"
        assert result["sggu_name"] == "강남구"

    def test_ambiguous_sggu_composite_key(self):
        """중복 시군구 복합 키 → 올바른 sgguCd"""
        result = parse_region("서울 중구")
        assert result["sgguCd"] == "110017"
        assert result["sido_name"] == "서울특별시"

    def test_ambiguous_sggu_different_sido(self):
        """다른 시도의 중복 시군구 → 다른 sgguCd"""
        result = parse_region("부산 동구")
        assert result["sgguCd"] == "210002"
        assert result["sido_name"] == "부산광역시"

    def test_full_sido_name_with_sggu(self):
        """정식 시도명 + 시군구 → 올바른 매핑"""
        result = parse_region("서울특별시 종로구")
        assert result["sgguCd"] == "110016"
        assert result["sido_name"] == "서울특별시"

    def test_fallback_unknown_region(self):
        """알 수 없는 지역명 → 폴백 (빈 코드, raw 보존)"""
        result = parse_region("알수없는곳")
        assert result["sidoCd"] == ""
        assert result["sgguCd"] == ""
        assert result["raw"] == "알수없는곳"

    def test_sejong(self):
        """세종시 특수 케이스"""
        result = parse_region("세종")
        assert result["sidoCd"] == "410000"
        assert result["sido_name"] == "세종특별자치시"

    def test_sido_short_with_sggu(self):
        """축약 시도명 + 고유 시군구 → 올바른 매핑"""
        result = parse_region("서울 중랑구")
        assert result["sidoCd"] == "110000"
        assert result["sgguCd"] == "110019"
        assert result["sggu_name"] == "중랑구"

    def test_emdong_extraction(self):
        """시도 + 시군구 + 읍면동 → 읍면동 추출"""
        result = parse_region("서울 중랑구 중화동")
        assert result["sidoCd"] == "110000"
        assert result["sgguCd"] == "110019"
        assert result["emdongNm"] == "중화동"

    def test_no_emdong(self):
        """읍면동 없는 경우 → emdongNm 빈 문자열"""
        result = parse_region("강남구")
        assert result["emdongNm"] == ""

    def test_sggu_with_emdong_no_sido(self):
        """시도 없이 시군구 + 읍면동 → 올바른 매핑"""
        result = parse_region("중랑구 중화동")
        assert result["sidoCd"] == "110000"
        assert result["sgguCd"] == "110019"
        assert result["emdongNm"] == "중화동"
