"""전국 시도/시군구 코드 매핑 및 지역명 파싱 유틸리티.

건강보험심사평가원 API(숫자 코드)와 국립중앙의료원 응급실 API(한글 시도명)를
동시에 지원하기 위해, 하나의 지역명 입력에서 두 가지 형식을 모두 반환한다.
"""

# 중복 시군구 목록 (여러 시도에 동일 이름 존재)
# 이 이름들은 단독 키로 등록하지 않고, "시도 시군구" 복합 키로만 등록
_AMBIGUOUS_NAMES = {"중구", "동구", "서구", "남구", "북구", "강서구"}

_REGION_DB: dict[str, dict] = {
    # =========================================
    # 시도 레벨 (17개)
    # =========================================
    "서울": {"sidoCd": "110000", "sgguCd": "", "sido_name": "서울특별시", "sggu_name": ""},
    "부산": {"sidoCd": "210000", "sgguCd": "", "sido_name": "부산광역시", "sggu_name": ""},
    "인천": {"sidoCd": "220000", "sgguCd": "", "sido_name": "인천광역시", "sggu_name": ""},
    "대구": {"sidoCd": "230000", "sgguCd": "", "sido_name": "대구광역시", "sggu_name": ""},
    "광주": {"sidoCd": "240000", "sgguCd": "", "sido_name": "광주광역시", "sggu_name": ""},
    "대전": {"sidoCd": "250000", "sgguCd": "", "sido_name": "대전광역시", "sggu_name": ""},
    "울산": {"sidoCd": "260000", "sgguCd": "", "sido_name": "울산광역시", "sggu_name": ""},
    "경기": {"sidoCd": "310000", "sgguCd": "", "sido_name": "경기도", "sggu_name": ""},
    "강원": {"sidoCd": "320000", "sgguCd": "", "sido_name": "강원특별자치도", "sggu_name": ""},
    "충북": {"sidoCd": "330000", "sgguCd": "", "sido_name": "충청북도", "sggu_name": ""},
    "충남": {"sidoCd": "340000", "sgguCd": "", "sido_name": "충청남도", "sggu_name": ""},
    "전북": {"sidoCd": "350000", "sgguCd": "", "sido_name": "전북특별자치도", "sggu_name": ""},
    "전남": {"sidoCd": "360000", "sgguCd": "", "sido_name": "전라남도", "sggu_name": ""},
    "경북": {"sidoCd": "370000", "sgguCd": "", "sido_name": "경상북도", "sggu_name": ""},
    "경남": {"sidoCd": "380000", "sgguCd": "", "sido_name": "경상남도", "sggu_name": ""},
    "제주": {"sidoCd": "390000", "sgguCd": "", "sido_name": "제주특별자치도", "sggu_name": ""},
    "세종": {"sidoCd": "410000", "sgguCd": "410000", "sido_name": "세종특별자치시", "sggu_name": "세종시"},

    # =========================================
    # 서울특별시 (24개 구) — 강서구/중구는 복합 키만
    # =========================================
    "강남구": {"sidoCd": "110000", "sgguCd": "110001", "sido_name": "서울특별시", "sggu_name": "강남구"},
    "강동구": {"sidoCd": "110000", "sgguCd": "110002", "sido_name": "서울특별시", "sggu_name": "강동구"},
    "관악구": {"sidoCd": "110000", "sgguCd": "110004", "sido_name": "서울특별시", "sggu_name": "관악구"},
    "광진구": {"sidoCd": "110000", "sgguCd": "110023", "sido_name": "서울특별시", "sggu_name": "광진구"},
    "구로구": {"sidoCd": "110000", "sgguCd": "110005", "sido_name": "서울특별시", "sggu_name": "구로구"},
    "금천구": {"sidoCd": "110000", "sgguCd": "110025", "sido_name": "서울특별시", "sggu_name": "금천구"},
    "노원구": {"sidoCd": "110000", "sgguCd": "110022", "sido_name": "서울특별시", "sggu_name": "노원구"},
    "도봉구": {"sidoCd": "110000", "sgguCd": "110006", "sido_name": "서울특별시", "sggu_name": "도봉구"},
    "동대문구": {"sidoCd": "110000", "sgguCd": "110007", "sido_name": "서울특별시", "sggu_name": "동대문구"},
    "동작구": {"sidoCd": "110000", "sgguCd": "110008", "sido_name": "서울특별시", "sggu_name": "동작구"},
    "마포구": {"sidoCd": "110000", "sgguCd": "110009", "sido_name": "서울특별시", "sggu_name": "마포구"},
    "서대문구": {"sidoCd": "110000", "sgguCd": "110010", "sido_name": "서울특별시", "sggu_name": "서대문구"},
    "서초구": {"sidoCd": "110000", "sgguCd": "110021", "sido_name": "서울특별시", "sggu_name": "서초구"},
    "성동구": {"sidoCd": "110000", "sgguCd": "110011", "sido_name": "서울특별시", "sggu_name": "성동구"},
    "성북구": {"sidoCd": "110000", "sgguCd": "110012", "sido_name": "서울특별시", "sggu_name": "성북구"},
    "송파구": {"sidoCd": "110000", "sgguCd": "110018", "sido_name": "서울특별시", "sggu_name": "송파구"},
    "양천구": {"sidoCd": "110000", "sgguCd": "110020", "sido_name": "서울특별시", "sggu_name": "양천구"},
    "영등포구": {"sidoCd": "110000", "sgguCd": "110013", "sido_name": "서울특별시", "sggu_name": "영등포구"},
    "용산구": {"sidoCd": "110000", "sgguCd": "110014", "sido_name": "서울특별시", "sggu_name": "용산구"},
    "은평구": {"sidoCd": "110000", "sgguCd": "110015", "sido_name": "서울특별시", "sggu_name": "은평구"},
    "종로구": {"sidoCd": "110000", "sgguCd": "110016", "sido_name": "서울특별시", "sggu_name": "종로구"},
    "중랑구": {"sidoCd": "110000", "sgguCd": "110019", "sido_name": "서울특별시", "sggu_name": "중랑구"},
    "강북구": {"sidoCd": "110000", "sgguCd": "110024", "sido_name": "서울특별시", "sggu_name": "강북구"},
    # 서울 중복 시군구
    "서울 중구": {"sidoCd": "110000", "sgguCd": "110017", "sido_name": "서울특별시", "sggu_name": "중구"},
    "서울 강서구": {"sidoCd": "110000", "sgguCd": "110003", "sido_name": "서울특별시", "sggu_name": "강서구"},

    # =========================================
    # 부산광역시 (16개 구/군)
    # =========================================
    "부산진구": {"sidoCd": "210000", "sgguCd": "210004", "sido_name": "부산광역시", "sggu_name": "부산진구"},
    "동래구": {"sidoCd": "210000", "sgguCd": "210003", "sido_name": "부산광역시", "sggu_name": "동래구"},
    "영도구": {"sidoCd": "210000", "sgguCd": "210007", "sido_name": "부산광역시", "sggu_name": "영도구"},
    "해운대구": {"sidoCd": "210000", "sgguCd": "210009", "sido_name": "부산광역시", "sggu_name": "해운대구"},
    "사하구": {"sidoCd": "210000", "sgguCd": "210010", "sido_name": "부산광역시", "sggu_name": "사하구"},
    "금정구": {"sidoCd": "210000", "sgguCd": "210011", "sido_name": "부산광역시", "sggu_name": "금정구"},
    "연제구": {"sidoCd": "210000", "sgguCd": "210013", "sido_name": "부산광역시", "sggu_name": "연제구"},
    "수영구": {"sidoCd": "210000", "sgguCd": "210014", "sido_name": "부산광역시", "sggu_name": "수영구"},
    "사상구": {"sidoCd": "210000", "sgguCd": "210015", "sido_name": "부산광역시", "sggu_name": "사상구"},
    "기장군": {"sidoCd": "210000", "sgguCd": "210100", "sido_name": "부산광역시", "sggu_name": "기장군"},
    # 부산 중복 시군구
    "부산 남구": {"sidoCd": "210000", "sgguCd": "210001", "sido_name": "부산광역시", "sggu_name": "남구"},
    "부산 동구": {"sidoCd": "210000", "sgguCd": "210002", "sido_name": "부산광역시", "sggu_name": "동구"},
    "부산 북구": {"sidoCd": "210000", "sgguCd": "210005", "sido_name": "부산광역시", "sggu_name": "북구"},
    "부산 서구": {"sidoCd": "210000", "sgguCd": "210006", "sido_name": "부산광역시", "sggu_name": "서구"},
    "부산 중구": {"sidoCd": "210000", "sgguCd": "210008", "sido_name": "부산광역시", "sggu_name": "중구"},
    "부산 강서구": {"sidoCd": "210000", "sgguCd": "210012", "sido_name": "부산광역시", "sggu_name": "강서구"},

    # =========================================
    # 인천광역시 (10개 구/군)
    # =========================================
    "미추홀구": {"sidoCd": "220000", "sgguCd": "220001", "sido_name": "인천광역시", "sggu_name": "미추홀구"},
    "부평구": {"sidoCd": "220000", "sgguCd": "220003", "sido_name": "인천광역시", "sggu_name": "부평구"},
    "남동구": {"sidoCd": "220000", "sgguCd": "220006", "sido_name": "인천광역시", "sggu_name": "남동구"},
    "연수구": {"sidoCd": "220000", "sgguCd": "220007", "sido_name": "인천광역시", "sggu_name": "연수구"},
    "계양구": {"sidoCd": "220000", "sgguCd": "220008", "sido_name": "인천광역시", "sggu_name": "계양구"},
    "강화군": {"sidoCd": "220000", "sgguCd": "220100", "sido_name": "인천광역시", "sggu_name": "강화군"},
    "옹진군": {"sidoCd": "220000", "sgguCd": "220200", "sido_name": "인천광역시", "sggu_name": "옹진군"},
    # 인천 중복 시군구
    "인천 동구": {"sidoCd": "220000", "sgguCd": "220002", "sido_name": "인천광역시", "sggu_name": "동구"},
    "인천 중구": {"sidoCd": "220000", "sgguCd": "220004", "sido_name": "인천광역시", "sggu_name": "중구"},
    "인천 서구": {"sidoCd": "220000", "sgguCd": "220005", "sido_name": "인천광역시", "sggu_name": "서구"},

    # =========================================
    # 대구광역시 (8개 구/군)
    # =========================================
    "수성구": {"sidoCd": "230000", "sgguCd": "230005", "sido_name": "대구광역시", "sggu_name": "수성구"},
    "달서구": {"sidoCd": "230000", "sgguCd": "230007", "sido_name": "대구광역시", "sggu_name": "달서구"},
    "달성군": {"sidoCd": "230000", "sgguCd": "230100", "sido_name": "대구광역시", "sggu_name": "달성군"},
    # 대구 중복 시군구
    "대구 남구": {"sidoCd": "230000", "sgguCd": "230001", "sido_name": "대구광역시", "sggu_name": "남구"},
    "대구 동구": {"sidoCd": "230000", "sgguCd": "230002", "sido_name": "대구광역시", "sggu_name": "동구"},
    "대구 북구": {"sidoCd": "230000", "sgguCd": "230003", "sido_name": "대구광역시", "sggu_name": "북구"},
    "대구 서구": {"sidoCd": "230000", "sgguCd": "230004", "sido_name": "대구광역시", "sggu_name": "서구"},
    "대구 중구": {"sidoCd": "230000", "sgguCd": "230006", "sido_name": "대구광역시", "sggu_name": "중구"},

    # =========================================
    # 광주광역시 (5개 구)
    # =========================================
    "광산구": {"sidoCd": "240000", "sgguCd": "240004", "sido_name": "광주광역시", "sggu_name": "광산구"},
    # 광주 중복 시군구
    "광주 동구": {"sidoCd": "240000", "sgguCd": "240001", "sido_name": "광주광역시", "sggu_name": "동구"},
    "광주 북구": {"sidoCd": "240000", "sgguCd": "240002", "sido_name": "광주광역시", "sggu_name": "북구"},
    "광주 서구": {"sidoCd": "240000", "sgguCd": "240003", "sido_name": "광주광역시", "sggu_name": "서구"},
    "광주 남구": {"sidoCd": "240000", "sgguCd": "240005", "sido_name": "광주광역시", "sggu_name": "남구"},

    # =========================================
    # 대전광역시 (5개 구)
    # =========================================
    "유성구": {"sidoCd": "250000", "sgguCd": "250001", "sido_name": "대전광역시", "sggu_name": "유성구"},
    "대덕구": {"sidoCd": "250000", "sgguCd": "250002", "sido_name": "대전광역시", "sggu_name": "대덕구"},
    # 대전 중복 시군구
    "대전 서구": {"sidoCd": "250000", "sgguCd": "250003", "sido_name": "대전광역시", "sggu_name": "서구"},
    "대전 동구": {"sidoCd": "250000", "sgguCd": "250004", "sido_name": "대전광역시", "sggu_name": "동구"},
    "대전 중구": {"sidoCd": "250000", "sgguCd": "250005", "sido_name": "대전광역시", "sggu_name": "중구"},

    # =========================================
    # 울산광역시 (5개 구/군)
    # =========================================
    "울주군": {"sidoCd": "260000", "sgguCd": "260100", "sido_name": "울산광역시", "sggu_name": "울주군"},
    # 울산 중복 시군구
    "울산 남구": {"sidoCd": "260000", "sgguCd": "260001", "sido_name": "울산광역시", "sggu_name": "남구"},
    "울산 동구": {"sidoCd": "260000", "sgguCd": "260002", "sido_name": "울산광역시", "sggu_name": "동구"},
    "울산 중구": {"sidoCd": "260000", "sgguCd": "260003", "sido_name": "울산광역시", "sggu_name": "중구"},
    "울산 북구": {"sidoCd": "260000", "sgguCd": "260004", "sido_name": "울산광역시", "sggu_name": "북구"},

    # =========================================
    # 경기도
    # =========================================
    "수원권선구": {"sidoCd": "310000", "sgguCd": "310601", "sido_name": "경기도", "sggu_name": "수원권선구"},
    "수원장안구": {"sidoCd": "310000", "sgguCd": "310602", "sido_name": "경기도", "sggu_name": "수원장안구"},
    "수원팔달구": {"sidoCd": "310000", "sgguCd": "310603", "sido_name": "경기도", "sggu_name": "수원팔달구"},
    "수원영통구": {"sidoCd": "310000", "sgguCd": "310604", "sido_name": "경기도", "sggu_name": "수원영통구"},
    "성남수정구": {"sidoCd": "310000", "sgguCd": "310401", "sido_name": "경기도", "sggu_name": "성남수정구"},
    "성남중원구": {"sidoCd": "310000", "sgguCd": "310402", "sido_name": "경기도", "sggu_name": "성남중원구"},
    "성남분당구": {"sidoCd": "310000", "sgguCd": "310403", "sido_name": "경기도", "sggu_name": "성남분당구"},
    "부천소사구": {"sidoCd": "310000", "sgguCd": "310301", "sido_name": "경기도", "sggu_name": "부천소사구"},
    "부천오정구": {"sidoCd": "310000", "sgguCd": "310302", "sido_name": "경기도", "sggu_name": "부천오정구"},
    "부천원미구": {"sidoCd": "310000", "sgguCd": "310303", "sido_name": "경기도", "sggu_name": "부천원미구"},
    "안양만안구": {"sidoCd": "310000", "sgguCd": "310701", "sido_name": "경기도", "sggu_name": "안양만안구"},
    "안양동안구": {"sidoCd": "310000", "sgguCd": "310702", "sido_name": "경기도", "sggu_name": "안양동안구"},
    "안산단원구": {"sidoCd": "310000", "sgguCd": "311101", "sido_name": "경기도", "sggu_name": "안산단원구"},
    "안산상록구": {"sidoCd": "310000", "sgguCd": "311102", "sido_name": "경기도", "sggu_name": "안산상록구"},
    "고양덕양구": {"sidoCd": "310000", "sgguCd": "311901", "sido_name": "경기도", "sggu_name": "고양덕양구"},
    "고양일산서구": {"sidoCd": "310000", "sgguCd": "311902", "sido_name": "경기도", "sggu_name": "고양일산서구"},
    "고양일산동구": {"sidoCd": "310000", "sgguCd": "311903", "sido_name": "경기도", "sggu_name": "고양일산동구"},
    "용인기흥구": {"sidoCd": "310000", "sgguCd": "312001", "sido_name": "경기도", "sggu_name": "용인기흥구"},
    "용인수지구": {"sidoCd": "310000", "sgguCd": "312002", "sido_name": "경기도", "sggu_name": "용인수지구"},
    "용인처인구": {"sidoCd": "310000", "sgguCd": "312003", "sido_name": "경기도", "sggu_name": "용인처인구"},
    "화성만세구": {"sidoCd": "310000", "sgguCd": "312501", "sido_name": "경기도", "sggu_name": "화성만세구"},
    "화성병점구": {"sidoCd": "310000", "sgguCd": "312503", "sido_name": "경기도", "sggu_name": "화성병점구"},
    "화성동탄구": {"sidoCd": "310000", "sgguCd": "312504", "sido_name": "경기도", "sggu_name": "화성동탄구"},
    "광명시": {"sidoCd": "310000", "sgguCd": "310100", "sido_name": "경기도", "sggu_name": "광명시"},
    "의정부시": {"sidoCd": "310000", "sgguCd": "310800", "sido_name": "경기도", "sggu_name": "의정부시"},
    "구리시": {"sidoCd": "310000", "sgguCd": "311000", "sido_name": "경기도", "sggu_name": "구리시"},
    "평택시": {"sidoCd": "310000", "sgguCd": "311200", "sido_name": "경기도", "sggu_name": "평택시"},
    "하남시": {"sidoCd": "310000", "sgguCd": "311300", "sido_name": "경기도", "sggu_name": "하남시"},
    "군포시": {"sidoCd": "310000", "sgguCd": "311400", "sido_name": "경기도", "sggu_name": "군포시"},
    "남양주시": {"sidoCd": "310000", "sgguCd": "311500", "sido_name": "경기도", "sggu_name": "남양주시"},
    "시흥시": {"sidoCd": "310000", "sgguCd": "311700", "sido_name": "경기도", "sggu_name": "시흥시"},
    "오산시": {"sidoCd": "310000", "sgguCd": "311800", "sido_name": "경기도", "sggu_name": "오산시"},
    "이천시": {"sidoCd": "310000", "sgguCd": "312100", "sido_name": "경기도", "sggu_name": "이천시"},
    "파주시": {"sidoCd": "310000", "sgguCd": "312200", "sido_name": "경기도", "sggu_name": "파주시"},
    "김포시": {"sidoCd": "310000", "sgguCd": "312300", "sido_name": "경기도", "sggu_name": "김포시"},
    "안성시": {"sidoCd": "310000", "sgguCd": "312400", "sido_name": "경기도", "sggu_name": "안성시"},
    "광주시": {"sidoCd": "310000", "sgguCd": "312600", "sido_name": "경기도", "sggu_name": "광주시"},
    "양주시": {"sidoCd": "310000", "sgguCd": "312700", "sido_name": "경기도", "sggu_name": "양주시"},
    "포천시": {"sidoCd": "310000", "sgguCd": "312800", "sido_name": "경기도", "sggu_name": "포천시"},
    "여주시": {"sidoCd": "310000", "sgguCd": "312900", "sido_name": "경기도", "sggu_name": "여주시"},
    "양평군": {"sidoCd": "310000", "sgguCd": "310009", "sido_name": "경기도", "sggu_name": "양평군"},

    # =========================================
    # 강원특별자치도
    # =========================================
    "강릉시": {"sidoCd": "320000", "sgguCd": "320100", "sido_name": "강원특별자치도", "sggu_name": "강릉시"},
    "동해시": {"sidoCd": "320000", "sgguCd": "320200", "sido_name": "강원특별자치도", "sggu_name": "동해시"},
    "속초시": {"sidoCd": "320000", "sgguCd": "320300", "sido_name": "강원특별자치도", "sggu_name": "속초시"},
    "원주시": {"sidoCd": "320000", "sgguCd": "320400", "sido_name": "강원특별자치도", "sggu_name": "원주시"},
    "춘천시": {"sidoCd": "320000", "sgguCd": "320500", "sido_name": "강원특별자치도", "sggu_name": "춘천시"},
    "태백시": {"sidoCd": "320000", "sgguCd": "320600", "sido_name": "강원특별자치도", "sggu_name": "태백시"},
    "삼척시": {"sidoCd": "320000", "sgguCd": "320700", "sido_name": "강원특별자치도", "sggu_name": "삼척시"},
    "양구군": {"sidoCd": "320000", "sgguCd": "320004", "sido_name": "강원특별자치도", "sggu_name": "양구군"},
    "영월군": {"sidoCd": "320000", "sgguCd": "320006", "sido_name": "강원특별자치도", "sggu_name": "영월군"},
    "인제군": {"sidoCd": "320000", "sgguCd": "320008", "sido_name": "강원특별자치도", "sggu_name": "인제군"},
    "정선군": {"sidoCd": "320000", "sgguCd": "320009", "sido_name": "강원특별자치도", "sggu_name": "정선군"},
    "철원군": {"sidoCd": "320000", "sgguCd": "320010", "sido_name": "강원특별자치도", "sggu_name": "철원군"},
    "평창군": {"sidoCd": "320000", "sgguCd": "320012", "sido_name": "강원특별자치도", "sggu_name": "평창군"},
    "홍천군": {"sidoCd": "320000", "sgguCd": "320013", "sido_name": "강원특별자치도", "sggu_name": "홍천군"},
    "화천군": {"sidoCd": "320000", "sgguCd": "320014", "sido_name": "강원특별자치도", "sggu_name": "화천군"},
    "횡성군": {"sidoCd": "320000", "sgguCd": "320015", "sido_name": "강원특별자치도", "sggu_name": "횡성군"},

    # =========================================
    # 충청북도
    # =========================================
    "청주상당구": {"sidoCd": "330000", "sgguCd": "330101", "sido_name": "충청북도", "sggu_name": "청주상당구"},
    "청주흥덕구": {"sidoCd": "330000", "sgguCd": "330102", "sido_name": "충청북도", "sggu_name": "청주흥덕구"},
    "청주청원구": {"sidoCd": "330000", "sgguCd": "330103", "sido_name": "충청북도", "sggu_name": "청주청원구"},
    "청주서원구": {"sidoCd": "330000", "sgguCd": "330104", "sido_name": "충청북도", "sggu_name": "청주서원구"},
    "충주시": {"sidoCd": "330000", "sgguCd": "330200", "sido_name": "충청북도", "sggu_name": "충주시"},
    "제천시": {"sidoCd": "330000", "sgguCd": "330300", "sido_name": "충청북도", "sggu_name": "제천시"},
    "괴산군": {"sidoCd": "330000", "sgguCd": "330001", "sido_name": "충청북도", "sggu_name": "괴산군"},
    "단양군": {"sidoCd": "330000", "sgguCd": "330002", "sido_name": "충청북도", "sggu_name": "단양군"},
    "보은군": {"sidoCd": "330000", "sgguCd": "330003", "sido_name": "충청북도", "sggu_name": "보은군"},
    "영동군": {"sidoCd": "330000", "sgguCd": "330004", "sido_name": "충청북도", "sggu_name": "영동군"},
    "옥천군": {"sidoCd": "330000", "sgguCd": "330005", "sido_name": "충청북도", "sggu_name": "옥천군"},
    "음성군": {"sidoCd": "330000", "sgguCd": "330006", "sido_name": "충청북도", "sggu_name": "음성군"},
    "진천군": {"sidoCd": "330000", "sgguCd": "330009", "sido_name": "충청북도", "sggu_name": "진천군"},
    "증평군": {"sidoCd": "330000", "sgguCd": "330011", "sido_name": "충청북도", "sggu_name": "증평군"},

    # =========================================
    # 충청남도
    # =========================================
    "천안서북구": {"sidoCd": "340000", "sgguCd": "340201", "sido_name": "충청남도", "sggu_name": "천안서북구"},
    "천안동남구": {"sidoCd": "340000", "sgguCd": "340202", "sido_name": "충청남도", "sggu_name": "천안동남구"},
    "공주시": {"sidoCd": "340000", "sgguCd": "340300", "sido_name": "충청남도", "sggu_name": "공주시"},
    "보령시": {"sidoCd": "340000", "sgguCd": "340400", "sido_name": "충청남도", "sggu_name": "보령시"},
    "아산시": {"sidoCd": "340000", "sgguCd": "340500", "sido_name": "충청남도", "sggu_name": "아산시"},
    "서산시": {"sidoCd": "340000", "sgguCd": "340600", "sido_name": "충청남도", "sggu_name": "서산시"},
    "논산시": {"sidoCd": "340000", "sgguCd": "340700", "sido_name": "충청남도", "sggu_name": "논산시"},
    "계룡시": {"sidoCd": "340000", "sgguCd": "340800", "sido_name": "충청남도", "sggu_name": "계룡시"},
    "당진시": {"sidoCd": "340000", "sgguCd": "340900", "sido_name": "충청남도", "sggu_name": "당진시"},
    "금산군": {"sidoCd": "340000", "sgguCd": "340002", "sido_name": "충청남도", "sggu_name": "금산군"},
    "부여군": {"sidoCd": "340000", "sgguCd": "340007", "sido_name": "충청남도", "sggu_name": "부여군"},
    "서천군": {"sidoCd": "340000", "sgguCd": "340009", "sido_name": "충청남도", "sggu_name": "서천군"},
    "예산군": {"sidoCd": "340000", "sgguCd": "340012", "sido_name": "충청남도", "sggu_name": "예산군"},
    "청양군": {"sidoCd": "340000", "sgguCd": "340014", "sido_name": "충청남도", "sggu_name": "청양군"},
    "홍성군": {"sidoCd": "340000", "sgguCd": "340015", "sido_name": "충청남도", "sggu_name": "홍성군"},

    # =========================================
    # 전북특별자치도
    # =========================================
    "전주완산구": {"sidoCd": "350000", "sgguCd": "350401", "sido_name": "전북특별자치도", "sggu_name": "전주완산구"},
    "전주덕진구": {"sidoCd": "350000", "sgguCd": "350402", "sido_name": "전북특별자치도", "sggu_name": "전주덕진구"},
    "군산시": {"sidoCd": "350000", "sgguCd": "350100", "sido_name": "전북특별자치도", "sggu_name": "군산시"},
    "남원시": {"sidoCd": "350000", "sgguCd": "350200", "sido_name": "전북특별자치도", "sggu_name": "남원시"},
    "익산시": {"sidoCd": "350000", "sgguCd": "350300", "sido_name": "전북특별자치도", "sggu_name": "익산시"},
    "정읍시": {"sidoCd": "350000", "sgguCd": "350500", "sido_name": "전북특별자치도", "sggu_name": "정읍시"},
    "김제시": {"sidoCd": "350000", "sgguCd": "350600", "sido_name": "전북특별자치도", "sggu_name": "김제시"},
    "고창군": {"sidoCd": "350000", "sgguCd": "350001", "sido_name": "전북특별자치도", "sggu_name": "고창군"},
    "무주군": {"sidoCd": "350000", "sgguCd": "350004", "sido_name": "전북특별자치도", "sggu_name": "무주군"},
    "부안군": {"sidoCd": "350000", "sgguCd": "350005", "sido_name": "전북특별자치도", "sggu_name": "부안군"},
    "순창군": {"sidoCd": "350000", "sgguCd": "350006", "sido_name": "전북특별자치도", "sggu_name": "순창군"},
    "완주군": {"sidoCd": "350000", "sgguCd": "350008", "sido_name": "전북특별자치도", "sggu_name": "완주군"},
    "진안군": {"sidoCd": "350000", "sgguCd": "350013", "sido_name": "전북특별자치도", "sggu_name": "진안군"},

    # =========================================
    # 전라남도
    # =========================================
    "목포시": {"sidoCd": "360000", "sgguCd": "360300", "sido_name": "전라남도", "sggu_name": "목포시"},
    "여수시": {"sidoCd": "360000", "sgguCd": "360500", "sido_name": "전라남도", "sggu_name": "여수시"},
    "순천시": {"sidoCd": "360000", "sgguCd": "360400", "sido_name": "전라남도", "sggu_name": "순천시"},
    "나주시": {"sidoCd": "360000", "sgguCd": "360200", "sido_name": "전라남도", "sggu_name": "나주시"},
    "광양시": {"sidoCd": "360000", "sgguCd": "360700", "sido_name": "전라남도", "sggu_name": "광양시"},
    "강진군": {"sidoCd": "360000", "sgguCd": "360001", "sido_name": "전라남도", "sggu_name": "강진군"},
    "고흥군": {"sidoCd": "360000", "sgguCd": "360002", "sido_name": "전라남도", "sggu_name": "고흥군"},
    "곡성군": {"sidoCd": "360000", "sgguCd": "360003", "sido_name": "전라남도", "sggu_name": "곡성군"},
    "구례군": {"sidoCd": "360000", "sgguCd": "360006", "sido_name": "전라남도", "sggu_name": "구례군"},
    "담양군": {"sidoCd": "360000", "sgguCd": "360008", "sido_name": "전라남도", "sggu_name": "담양군"},
    "무안군": {"sidoCd": "360000", "sgguCd": "360009", "sido_name": "전라남도", "sggu_name": "무안군"},
    "보성군": {"sidoCd": "360000", "sgguCd": "360010", "sido_name": "전라남도", "sggu_name": "보성군"},
    "신안군": {"sidoCd": "360000", "sgguCd": "360012", "sido_name": "전라남도", "sggu_name": "신안군"},
    "영광군": {"sidoCd": "360000", "sgguCd": "360014", "sido_name": "전라남도", "sggu_name": "영광군"},
    "영암군": {"sidoCd": "360000", "sgguCd": "360015", "sido_name": "전라남도", "sggu_name": "영암군"},
    "완도군": {"sidoCd": "360000", "sgguCd": "360016", "sido_name": "전라남도", "sggu_name": "완도군"},
    "장성군": {"sidoCd": "360000", "sgguCd": "360017", "sido_name": "전라남도", "sggu_name": "장성군"},
    "장흥군": {"sidoCd": "360000", "sgguCd": "360018", "sido_name": "전라남도", "sggu_name": "장흥군"},
    "진도군": {"sidoCd": "360000", "sgguCd": "360019", "sido_name": "전라남도", "sggu_name": "진도군"},
    "함평군": {"sidoCd": "360000", "sgguCd": "360020", "sido_name": "전라남도", "sggu_name": "함평군"},
    "해남군": {"sidoCd": "360000", "sgguCd": "360021", "sido_name": "전라남도", "sggu_name": "해남군"},
    "화순군": {"sidoCd": "360000", "sgguCd": "360022", "sido_name": "전라남도", "sggu_name": "화순군"},

    # =========================================
    # 경상북도
    # =========================================
    "포항남구": {"sidoCd": "370000", "sgguCd": "370701", "sido_name": "경상북도", "sggu_name": "포항남구"},
    "포항북구": {"sidoCd": "370000", "sgguCd": "370702", "sido_name": "경상북도", "sggu_name": "포항북구"},
    "경주시": {"sidoCd": "370000", "sgguCd": "370100", "sido_name": "경상북도", "sggu_name": "경주시"},
    "구미시": {"sidoCd": "370000", "sgguCd": "370200", "sido_name": "경상북도", "sggu_name": "구미시"},
    "김천시": {"sidoCd": "370000", "sgguCd": "370300", "sido_name": "경상북도", "sggu_name": "김천시"},
    "안동시": {"sidoCd": "370000", "sgguCd": "370400", "sido_name": "경상북도", "sggu_name": "안동시"},
    "영주시": {"sidoCd": "370000", "sgguCd": "370500", "sido_name": "경상북도", "sggu_name": "영주시"},
    "영천시": {"sidoCd": "370000", "sgguCd": "370600", "sido_name": "경상북도", "sggu_name": "영천시"},
    "문경시": {"sidoCd": "370000", "sgguCd": "370800", "sido_name": "경상북도", "sggu_name": "문경시"},
    "상주시": {"sidoCd": "370000", "sgguCd": "370900", "sido_name": "경상북도", "sggu_name": "상주시"},
    "경산시": {"sidoCd": "370000", "sgguCd": "371000", "sido_name": "경상북도", "sggu_name": "경산시"},
    "고령군": {"sidoCd": "370000", "sgguCd": "370002", "sido_name": "경상북도", "sggu_name": "고령군"},
    "봉화군": {"sidoCd": "370000", "sgguCd": "370007", "sido_name": "경상북도", "sggu_name": "봉화군"},
    "성주군": {"sidoCd": "370000", "sgguCd": "370010", "sido_name": "경상북도", "sggu_name": "성주군"},
    "영덕군": {"sidoCd": "370000", "sgguCd": "370012", "sido_name": "경상북도", "sggu_name": "영덕군"},
    "영양군": {"sidoCd": "370000", "sgguCd": "370013", "sido_name": "경상북도", "sggu_name": "영양군"},
    "예천군": {"sidoCd": "370000", "sgguCd": "370017", "sido_name": "경상북도", "sggu_name": "예천군"},
    "울진군": {"sidoCd": "370000", "sgguCd": "370019", "sido_name": "경상북도", "sggu_name": "울진군"},
    "의성군": {"sidoCd": "370000", "sgguCd": "370021", "sido_name": "경상북도", "sggu_name": "의성군"},
    "청도군": {"sidoCd": "370000", "sgguCd": "370022", "sido_name": "경상북도", "sggu_name": "청도군"},
    "칠곡군": {"sidoCd": "370000", "sgguCd": "370024", "sido_name": "경상북도", "sggu_name": "칠곡군"},

    # =========================================
    # 경상남도
    # =========================================
    "창원마산회원구": {"sidoCd": "380000", "sgguCd": "380701", "sido_name": "경상남도", "sggu_name": "창원마산회원구"},
    "창원마산합포구": {"sidoCd": "380000", "sgguCd": "380702", "sido_name": "경상남도", "sggu_name": "창원마산합포구"},
    "창원진해구": {"sidoCd": "380000", "sgguCd": "380703", "sido_name": "경상남도", "sggu_name": "창원진해구"},
    "창원의창구": {"sidoCd": "380000", "sgguCd": "380704", "sido_name": "경상남도", "sggu_name": "창원의창구"},
    "창원성산구": {"sidoCd": "380000", "sgguCd": "380705", "sido_name": "경상남도", "sggu_name": "창원성산구"},
    "김해시": {"sidoCd": "380000", "sgguCd": "380100", "sido_name": "경상남도", "sggu_name": "김해시"},
    "사천시": {"sidoCd": "380000", "sgguCd": "380300", "sido_name": "경상남도", "sggu_name": "사천시"},
    "진주시": {"sidoCd": "380000", "sgguCd": "380500", "sido_name": "경상남도", "sggu_name": "진주시"},
    "통영시": {"sidoCd": "380000", "sgguCd": "380800", "sido_name": "경상남도", "sggu_name": "통영시"},
    "밀양시": {"sidoCd": "380000", "sgguCd": "380900", "sido_name": "경상남도", "sggu_name": "밀양시"},
    "거제시": {"sidoCd": "380000", "sgguCd": "381000", "sido_name": "경상남도", "sggu_name": "거제시"},
    "양산시": {"sidoCd": "380000", "sgguCd": "381100", "sido_name": "경상남도", "sggu_name": "양산시"},
    "거창군": {"sidoCd": "380000", "sgguCd": "380002", "sido_name": "경상남도", "sggu_name": "거창군"},
    "고성군": {"sidoCd": "380000", "sgguCd": "380003", "sido_name": "경상남도", "sggu_name": "고성군"},
    "의령군": {"sidoCd": "380000", "sgguCd": "380011", "sido_name": "경상남도", "sggu_name": "의령군"},
    "함안군": {"sidoCd": "380000", "sgguCd": "380017", "sido_name": "경상남도", "sggu_name": "함안군"},
    "합천군": {"sidoCd": "380000", "sgguCd": "380019", "sido_name": "경상남도", "sggu_name": "합천군"},

    # =========================================
    # 제주특별자치도
    # =========================================
    "제주시": {"sidoCd": "390000", "sgguCd": "390200", "sido_name": "제주특별자치도", "sggu_name": "제주시"},
    "서귀포시": {"sidoCd": "390000", "sgguCd": "390100", "sido_name": "제주특별자치도", "sggu_name": "서귀포시"},
}


def parse_region(region: str) -> dict:
    """지역명을 파싱하여 API별 파라미터 반환.

    처리 순서:
    1. 원문 전체 매칭 ("서울 중구" → 복합 키 직접 매칭)
    2. 공백 분리 후 "시도 + 시군구" 복합 키 매칭 ("서울 강남구")
    3. 단독 키 매칭 ("강남구", "서울")
    4. 폴백 — 원문 그대로 반환
    """
    fallback = {"sidoCd": "", "sgguCd": "", "sido_name": "", "sggu_name": "", "emdongNm": "", "raw": region}

    def _result(db_entry, emdong: str = ""):
        return {**db_entry, "emdongNm": emdong, "raw": region}

    # 읍면동 추출 (3번째 파트가 있으면)
    parts = region.split()
    emdong = parts[2] if len(parts) >= 3 else ""

    # 1. 원문 전체 매칭
    if region in _REGION_DB:
        return _result(_REGION_DB[region], emdong)

    # 2. 공백 분리 → 복합 키 시도
    if len(parts) >= 2:
        composite = f"{parts[0]} {parts[1]}"
        if composite in _REGION_DB:
            return _result(_REGION_DB[composite], emdong)

        # "서울특별시 강남구" 또는 "서울 중랑구" → 시도명 매칭 후 복합 키 재시도
        for sido_short, data in _REGION_DB.items():
            if (data.get("sido_name") == parts[0] or sido_short == parts[0]) and not data.get("sgguCd"):
                composite2 = f"{sido_short} {parts[1]}"
                if composite2 in _REGION_DB:
                    return _result(_REGION_DB[composite2], emdong)
                if parts[1] in _REGION_DB:
                    return _result(_REGION_DB[parts[1]], emdong)
                break

        # "중랑구 중화동" → parts[0]이 시군구, parts[1]이 읍면동
        if parts[0] in _REGION_DB and _REGION_DB[parts[0]].get("sgguCd"):
            return _result(_REGION_DB[parts[0]], emdong=parts[1])

    # 3. 폴백
    return fallback
