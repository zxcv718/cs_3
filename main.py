from __future__ import annotations

"""
Mini NPU Simulator

이 파일은 콘솔에서 실행하는 간단한 NPU 시뮬레이터다.
사용자는 두 가지 방식으로 프로그램을 실행할 수 있다.

1. 3x3 필터와 패턴을 직접 입력해서 MAC 결과를 확인하는 모드
2. data.json 파일을 읽어서 여러 테스트 케이스를 자동 분석하는 모드

핵심 개념은 MAC(Multiply-Accumulate) 연산이다.
같은 위치의 값끼리 곱한 뒤 전부 더해 MAC 점수를 만들고,
그 점수가 입력 패턴의 총합과 얼마나 가까운지 절대오차로 바꿔 최종 판정한다.

이 파일은 "동작" 자체보다 "읽고 이해하기 쉬운 구조"도 중요하게 생각해서 작성되어 있다.
그래서 함수 역할을 잘게 나누고, JSON 검증/라벨 정규화/판정/성능 측정을 각각 분리해 두었다.
"""

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Matrix는 "2차원 숫자 배열"을 뜻한다.
# 예를 들어 3x3 패턴이라면 [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]] 같은 형태다.
Matrix = List[List[float]]

# 두 거리 값을 비교할 때 사용할 허용 오차다.
# 부동소수점은 사람이 기대하는 값과 아주 조금 다르게 저장될 수 있어서,
# 완전히 같지 않더라도 차이가 매우 작으면 "동점"으로 보는 정책이 필요하다.
EPSILON = 1e-9

# 성능 측정은 한 번만 재면 우연한 오차가 섞일 수 있다.
# 그래서 같은 연산을 10번 반복한 뒤 평균을 사용한다.
BENCHMARK_REPEATS = 10

# JSON 모드에서 공식적으로 다루는 필터 크기 목록이다.
JSON_SIZES = (5, 13, 25)

# patterns의 키는 반드시 size_{N}_{idx} 형식이라고 가정한다.
# 예: size_5_1, size_13_2
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")


@dataclass
class JsonCase:
    """JSON의 patterns 항목 하나를 해석한 결과를 담는 자료형."""

    id: str
    size: int
    pattern: Matrix
    expected_label: str


@dataclass
class CaseResult:
    """
    테스트 케이스 1개를 평가한 뒤의 결과를 담는다.

    score_cross / score_x:
        각 필터와의 MAC 점수. 계산 자체가 불가능한 경우(None)일 수 있다.
    distance_cross / distance_x:
        각 필터의 MAC 결과가 입력 패턴 총합에서 얼마나 벗어났는지 나타내는 절대오차.
        계산 자체가 불가능한 경우(None)일 수 있다.
    predicted:
        프로그램이 최종적으로 내린 판정값(Cross, X, UNDECIDED, N/A)
    expected:
        JSON에 적혀 있던 정답 라벨. 읽을 수 없었던 경우 None일 수 있다.
    status:
        PASS 또는 FAIL
    reason:
        왜 PASS/FAIL이 되었는지 사람이 읽을 수 있게 남기는 설명
    """

    id: str
    score_cross: Optional[float]
    score_x: Optional[float]
    distance_cross: Optional[float]
    distance_x: Optional[float]
    predicted: str
    expected: Optional[str]
    status: str
    reason: str


@dataclass
class PerformanceRow:
    """성능 분석 표의 한 줄을 표현하는 자료형."""

    size: int
    average_ms: Optional[float]
    operation_count: int


def print_banner() -> None:
    """프로그램 시작 시 맨 위에 출력할 제목을 보여준다."""
    print("=== Mini NPU Simulator ===")


def print_section(index: int, title: str) -> None:
    """콘솔 출력을 보기 쉽게 섹션 단위로 구분한다."""
    print()
    print("#---------------------------------------")
    print(f"# [{index}] {title}")
    print("#---------------------------------------")


def prompt_mode() -> str:
    """
    사용자에게 실행 모드를 물어본다.

    유효한 입력은 "1" 또는 "2" 뿐이며,
    잘못 입력하면 다시 물어본다.
    """
    while True:
        print()
        print("[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        choice = input("선택: ").strip()
        if choice in {"1", "2"}:
            return choice
        print("입력 오류: 1 또는 2를 입력하세요.")


def format_cell(value: float) -> str:
    """
    행렬을 화면에 예쁘게 출력하기 위한 보조 함수다.

    예를 들어 1.0은 굳이 1.0으로 보이지 않고 1로 보이게 해서
    초심자가 출력값을 더 편하게 읽을 수 있도록 한다.
    """
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def print_matrix(matrix: Matrix) -> None:
    """2차원 행렬을 사람이 읽기 좋은 형태로 한 줄씩 출력한다."""
    for row in matrix:
        print(" ".join(format_cell(value) for value in row))


def format_score(value: Optional[float]) -> str:
    """점수 출력용 포맷 함수. 점수가 없으면 N/A를 출력한다."""
    if value is None:
        return "N/A"
    return repr(value)


def format_distance(value: Optional[float]) -> str:
    """거리 출력용 포맷 함수. 계산할 수 없으면 N/A를 출력한다."""
    if value is None:
        return "N/A"
    return repr(value)


def format_ms(value: Optional[float]) -> str:
    """밀리초(ms) 값을 소수점 6자리까지 맞춰 문자열로 바꾼다."""
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def normalize_label(raw: Any) -> str:
    """
    다양한 입력 라벨을 프로그램 내부의 표준 라벨로 통일한다.

    왜 필요한가?
    - JSON expected 값은 '+' 또는 'x'일 수 있다.
    - 필터 키는 'cross' 또는 'x'일 수 있다.
    - 하지만 프로그램 내부에서는 비교 기준이 하나로 고정되어야 실수가 줄어든다.

    그래서 내부에서는 오직 다음 두 값만 사용한다.
    - Cross
    - X
    """
    if not isinstance(raw, str):
        raise ValueError("라벨은 문자열이어야 합니다.")

    normalized = raw.strip().lower()
    if normalized in {"+", "cross"}:
        return "Cross"
    if normalized == "x":
        return "X"
    raise ValueError(f"지원하지 않는 라벨입니다: {raw}")


def is_finite_number(value: float) -> bool:
    """
    NaN, Infinity 같은 비정상 실수를 걸러내기 위한 보조 함수다.

    거리 기반 판정은 "유한한 실수"를 전제로 해야 의미가 있으므로,
    입력 단계에서 이런 값을 미리 막아 두는 편이 안전하다.
    """
    return math.isfinite(value)


def validate_matrix(matrix_data: Any, expected_size: int) -> Tuple[Optional[Matrix], str]:
    """
    입력값이 "expected_size x expected_size" 형태의 숫자 2차원 배열인지 검사한다.

    반환 규칙:
    - 성공: (검증된 Matrix, "")
    - 실패: (None, 실패 사유 메시지)

    예외를 바로 터뜨리지 않고 실패 사유를 문자열로 돌려주는 이유는,
    JSON 모드에서 특정 케이스만 FAIL 처리하고 프로그램 전체는 계속 진행하기 위해서다.
    """
    if not isinstance(matrix_data, list):
        return None, "schema_error: 2차원 배열(list)이 아닙니다."
    if len(matrix_data) != expected_size:
        return None, (
            f"size_mismatch: 행 수가 {expected_size}가 아닙니다. "
            f"(현재 {len(matrix_data)})"
        )

    validated: Matrix = []
    for row_index, row in enumerate(matrix_data, start=1):
        if not isinstance(row, list):
            return None, f"schema_error: {row_index}번째 행이 배열이 아닙니다."
        if len(row) != expected_size:
            return None, (
                f"size_mismatch: {row_index}번째 행의 열 수가 {expected_size}가 아닙니다. "
                f"(현재 {len(row)})"
            )

        converted_row: List[float] = []
        for col_index, value in enumerate(row, start=1):
            # bool은 int의 하위 타입처럼 보일 수 있지만,
            # 여기서는 진짜 숫자 데이터만 허용하고 싶어서 따로 막아 둔다.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, (
                    f"schema_error: {row_index}행 {col_index}열 값이 숫자가 아닙니다."
                )
            converted_value = float(value)
            if not is_finite_number(converted_value):
                return None, (
                    f"schema_error: {row_index}행 {col_index}열 값이 "
                    "유한한 숫자(NaN/Infinity 제외)가 아닙니다."
                )
            converted_row.append(converted_value)
        validated.append(converted_row)

    return validated, ""


def parse_console_row(raw: str, expected_size: int) -> Tuple[Optional[List[float]], str]:
    """
    콘솔에서 입력한 "한 줄"을 숫자 배열 1행으로 바꾼다.

    예: "0 1 0" -> [0.0, 1.0, 0.0]

    여기서는 두 가지만 최소 검증한다.
    - 숫자 개수가 expected_size와 맞는가?
    - 각 항목이 실제로 숫자로 변환 가능한가?
    """
    parts = raw.strip().split()
    if len(parts) != expected_size:
        return None, (
            f"입력 형식 오류: 각 줄에 {expected_size}개의 숫자를 "
            "공백으로 구분해 입력하세요."
        )

    values: List[float] = []
    for part in parts:
        try:
            converted_value = float(part)
        except ValueError:
            return None, (
                f"입력 형식 오류: 각 줄에 {expected_size}개의 숫자를 "
                "공백으로 구분해 입력하세요."
            )
        if not is_finite_number(converted_value):
            return None, (
                "입력 형식 오류: NaN 또는 Infinity가 아닌 "
                f"{expected_size}개의 유한한 숫자를 입력하세요."
            )
        values.append(converted_value)
    return values, ""


def prompt_matrix(name: str, size: int) -> Matrix:
    """
    size x size 행렬 전체를 사용자에게 직접 입력받는다.

    핵심 포인트는 "행 단위 재입력"이다.
    예를 들어 2행만 잘못 입력했다면 1행부터 다시 받을 필요 없이
    2행만 다시 입력하게 해서 사용성을 높였다.
    """
    print(f"{name} ({size}줄 입력, 공백 구분)")
    rows: Matrix = []
    for row_index in range(1, size + 1):
        while True:
            raw = input(f"{row_index}행: ")
            row, error = parse_console_row(raw, size)
            if row is not None:
                rows.append(row)
                break
            print(error)
    return rows


def compute_mac(pattern: Matrix, filter_matrix: Matrix) -> float:
    """
    MAC(Multiply-Accumulate) 연산의 핵심 함수다.

    동작 방식:
    1. pattern과 filter_matrix의 같은 위치 값을 하나씩 꺼낸다.
    2. 두 값을 곱한다.
    3. 그 결과를 total에 계속 더한다.
    4. 모든 칸을 다 돌면 total이 최종 점수다.

    예를 들어 3x3 행렬이면 총 9개의 위치를 방문한다.
    N x N 행렬이면 총 N^2번의 위치 비교가 일어난다.
    """
    total = 0.0
    for row_index in range(len(pattern)):
        for col_index in range(len(pattern[row_index])):
            total += pattern[row_index][col_index] * filter_matrix[row_index][col_index]
    return total


def compute_pattern_sum(pattern: Matrix) -> float:
    """
    입력 패턴의 총합을 계산한다.

    현재 판정 정책은 "필터의 기준점수"를 따로 두지 않고,
    MAC 결과가 입력 패턴 총합에 얼마나 가까운지를 본다.
    즉 같은 패턴 하나를 두 필터와 각각 비교한 뒤,
    두 결과가 "패턴이 가진 전체 값"을 얼마나 잘 설명하는지 비교한다.

    예:
    - pattern 합이 5이고 MAC 결과가 5면 오차는 0
    - pattern 합이 5이고 MAC 결과가 50이면 오차는 45
    """
    total = 0.0
    for row in pattern:
        for value in row:
            total += value
    return total


def compute_distance(score: float, pattern_sum: float) -> float:
    """
    MAC 점수가 입력 패턴 총합에서 얼마나 떨어져 있는지 계산한다.

    절대오차가 0에 가까울수록 그 필터가 패턴과 더 잘 맞는다고 해석한다.
    """
    return abs(score - pattern_sum)


def classify_distances(
    distance_cross: float,
    distance_x: float,
    epsilon: float = EPSILON,
) -> str:
    """
    Cross 거리와 X 거리를 비교해서 최종 라벨을 정한다.

    규칙:
    - 두 거리 차이가 epsilon보다 작으면 UNDECIDED
    - Cross 거리가 더 작으면 Cross
    - X 거리가 더 작으면 X
    """
    if abs(distance_cross - distance_x) < epsilon:
        return "UNDECIDED"
    if distance_cross < distance_x:
        return "Cross"
    return "X"


def benchmark_mac(pattern: Matrix, filter_matrix: Matrix, repeats: int = BENCHMARK_REPEATS) -> float:
    """
    MAC 연산 성능을 측정한다.

    중요한 점:
    - 입력/출력/JSON 로드 시간은 재지 않는다.
    - 순수하게 compute_mac() 호출 구간만 측정한다.
    - 여러 번 반복해서 평균값을 사용한다.
    """
    total_seconds = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        compute_mac(pattern, filter_matrix)
        total_seconds += time.perf_counter() - start
    return (total_seconds / repeats) * 1000.0


def print_performance_table(rows: List[PerformanceRow]) -> None:
    """성능 측정 결과를 표처럼 정렬해서 출력한다."""
    print("크기(N×N)   평균 시간(ms)    연산 횟수(N²)")
    print("-------------------------------------------")
    for row in rows:
        size_label = f"{row.size}×{row.size}"
        avg_label = format_ms(row.average_ms)
        print(f"{size_label:<12}{avg_label:>14}{row.operation_count:>16}")


def default_cross_3x3() -> Matrix:
    """성능 분석용 기본 3x3 Cross 패턴을 반환한다."""
    return [
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ]


def load_json_root(json_path: Path) -> Optional[Dict[str, Any]]:
    """
    data.json 파일을 읽고, 최상위 구조가 최소 조건을 만족하는지 확인한다.

    여기서 확인하는 것:
    - 파일이 실제로 존재하는가?
    - JSON 파싱이 가능한가?
    - 최상위가 dict인가?
    - filters, patterns가 모두 dict인가?
    """
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"JSON 로드 오류: {json_path.name} 파일을 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError as error:
        print(f"JSON 로드 오류: JSON 파싱에 실패했습니다. ({error})")
        return None
    except OSError as error:
        print(f"JSON 로드 오류: 파일을 읽는 중 문제가 발생했습니다. ({error})")
        return None

    if not isinstance(data, dict):
        print("JSON 로드 오류: 최상위 구조는 객체(dict)여야 합니다.")
        return None

    filters = data.get("filters")
    patterns = data.get("patterns")
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        print("JSON 로드 오류: top-level에 filters 와 patterns 객체가 필요합니다.")
        return None

    return data


def load_filters(
    filters_data: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Matrix]], List[str], Dict[int, List[str]]]:
    """
    JSON의 filters 영역을 읽어서 "크기별 필터 사전"으로 정리한다.

    반환값 3개는 각각 다음 의미를 가진다.
    1. filters_by_size:
       실제 계산에 사용할 정상 필터들
       예: {5: {"Cross": [...], "X": [...]}, 13: {...}}
    2. messages:
       콘솔에 바로 출력할 로드 결과 메시지
    3. issues_by_size:
       각 크기에서 발견된 문제 목록

    이렇게 분리해 두면,
    - 화면 출력용 정보
    - 실제 계산용 정보
    - 오류 원인 추적용 정보
    를 서로 깔끔하게 관리할 수 있다.
    """
    filters_by_size: Dict[int, Dict[str, Matrix]] = {}
    messages: List[str] = []
    issues_by_size: Dict[int, List[str]] = {}

    for size in JSON_SIZES:
        size_key = f"size_{size}"
        raw_bucket = filters_data.get(size_key)
        issues: List[str] = []
        valid_filters: Dict[str, Matrix] = {}

        if not isinstance(raw_bucket, dict):
            issues.append("필터 버킷이 없거나 객체가 아닙니다.")
        else:
            for raw_key, matrix_data in raw_bucket.items():
                # raw_key가 cross, x처럼 제각각 들어와도
                # 내부에서는 Cross/X라는 표준 라벨만 쓰기 위해 정규화한다.
                try:
                    label = normalize_label(raw_key)
                except ValueError:
                    # 지원하지 않는 필터 키는 무시한다.
                    # 대신 뒤에서 "필수 필터가 없다"는 형태로 문제를 드러낸다.
                    continue

                if label in valid_filters:
                    issues.append(f"{label} 필터가 중복되었습니다.")
                    continue

                validated_matrix, error = validate_matrix(matrix_data, size)
                if validated_matrix is None:
                    issues.append(f"{label} 필터 오류: {error}")
                    continue
                valid_filters[label] = validated_matrix

        for required_label in ("Cross", "X"):
            # 계산하려면 Cross 필터와 X 필터가 모두 필요하다.
            if required_label not in valid_filters:
                issues.append(f"{required_label} 필터가 없습니다.")

        if valid_filters:
            filters_by_size[size] = valid_filters
        if issues:
            issues_by_size[size] = issues.copy()

        if issues:
            messages.append(f"✗ {size_key} 필터 로드 실패: {'; '.join(issues)}")
        else:
            messages.append(f"✓ {size_key:<7} 필터 로드 완료 (Cross, X)")

    return filters_by_size, messages, issues_by_size


def parse_pattern_size(case_id: str) -> Optional[int]:
    """
    케이스 ID에서 크기 N을 뽑아낸다.

    예:
    - size_5_1  -> 5
    - size_13_2 -> 13
    """
    match = PATTERN_KEY_RE.match(case_id)
    if match is None:
        return None
    return int(match.group(1))


def build_failure_result(
    case_id: str,
    reason: str,
    expected: Optional[str] = None,
) -> CaseResult:
    """
    계산 불가능한 케이스를 FAIL 결과 객체로 빠르게 만드는 헬퍼 함수다.

    JSON 검증 실패, 필터 누락, 크기 불일치 같은 경우에 사용한다.
    """
    return CaseResult(
        id=case_id,
        score_cross=None,
        score_x=None,
        distance_cross=None,
        distance_x=None,
        predicted="N/A",
        expected=expected,
        status="FAIL",
        reason=reason,
    )


def parse_json_case(case_id: str, case_data: Any) -> Tuple[Optional[JsonCase], Optional[CaseResult]]:
    """
    JSON의 patterns 항목 1개를 읽어 실제 평가 가능한 JsonCase로 바꾼다.

    성공하면:
    - (JsonCase, None)

    실패하면:
    - (None, FAIL용 CaseResult)

    즉, 이 함수는 "파싱 + 1차 검증" 단계라고 생각하면 된다.
    """
    size = parse_pattern_size(case_id)
    if size is None:
        return None, build_failure_result(
            case_id,
            "schema_error: 케이스 키는 size_{N}_{idx} 형식이어야 합니다.",
        )

    if not isinstance(case_data, dict):
        return None, build_failure_result(
            case_id,
            "schema_error: 패턴 항목이 객체(dict)가 아닙니다.",
        )

    if "input" not in case_data:
        return None, build_failure_result(
            case_id,
            "schema_error: input 필드가 없습니다.",
        )
    if "expected" not in case_data:
        return None, build_failure_result(
            case_id,
            "schema_error: expected 필드가 없습니다.",
        )

    # input은 size x size 숫자 행렬이어야 한다.
    pattern, matrix_error = validate_matrix(case_data["input"], size)
    if pattern is None:
        return None, build_failure_result(case_id, matrix_error)

    try:
        # expected도 표준 라벨(Cross/X)로 통일해서 이후 비교를 단순하게 만든다.
        expected_label = normalize_label(case_data["expected"])
    except ValueError as error:
        return None, build_failure_result(
            case_id,
            f"invalid_label: {error}",
        )

    return JsonCase(case_id, size, pattern, expected_label), None


def evaluate_case(
    case: JsonCase,
    filters_by_size: Dict[int, Dict[str, Matrix]],
    filter_issues_by_size: Dict[int, List[str]],
) -> CaseResult:
    """
    검증을 통과한 JsonCase 1개를 실제로 평가한다.

    처리 순서:
    1. 같은 크기의 Cross/X 필터를 찾는다.
    2. 둘 중 하나라도 없으면 FAIL 처리한다.
    3. 두 MAC 점수를 계산한다.
    4. 입력 패턴 총합을 기준으로 각 필터의 절대오차를 계산한다.
    5. 절대오차를 비교해 Cross/X/UNDECIDED를 판정한다.
    6. expected와 비교해 PASS/FAIL을 결정한다.
    """
    size_filters = filters_by_size.get(case.size, {})
    cross_filter = size_filters.get("Cross")
    x_filter = size_filters.get("X")

    if cross_filter is None or x_filter is None:
        # 필터가 없는 이유를 좀 더 구체적으로 설명해 주기 위해
        # 이미 수집해 둔 issues 정보를 참고한다.
        issue_messages = filter_issues_by_size.get(case.size, [])
        if any("size_mismatch" in issue for issue in issue_messages):
            reason = (
                f"size_mismatch: size_{case.size} 필터의 크기가 "
                "패턴 크기와 일치하지 않습니다."
            )
        else:
            reason = (
                f"missing_filter: size_{case.size}의 Cross/X 필터를 "
                "모두 찾을 수 없습니다."
            )
        return build_failure_result(
            case.id,
            reason,
            expected=case.expected_label,
        )

    # 같은 패턴을 두 필터와 각각 비교해서 점수를 계산한다.
    score_cross = compute_mac(case.pattern, cross_filter)
    score_x = compute_mac(case.pattern, x_filter)
    pattern_sum = compute_pattern_sum(case.pattern)
    distance_cross = compute_distance(score_cross, pattern_sum)
    distance_x = compute_distance(score_x, pattern_sum)
    predicted = classify_distances(distance_cross, distance_x)

    # predicted는 프로그램의 판정,
    # expected_label은 데이터셋이 기대한 정답이다.
    if predicted == case.expected_label:
        status = "PASS"
        reason = "matched_expected"
    elif predicted == "UNDECIDED":
        status = "FAIL"
        reason = f"undecided_by_epsilon: |distance_cross-distance_x| < {EPSILON}"
    else:
        status = "FAIL"
        reason = (
            f"prediction_mismatch: predicted {predicted}, "
            f"expected {case.expected_label}"
        )

    return CaseResult(
        id=case.id,
        score_cross=score_cross,
        score_x=score_x,
        distance_cross=distance_cross,
        distance_x=distance_x,
        predicted=predicted,
        expected=case.expected_label,
        status=status,
        reason=reason,
    )


def print_case_result(result: CaseResult) -> None:
    """케이스 1개의 평가 결과를 콘솔에 보기 좋게 출력한다."""
    print(f"--- {result.id} ---")
    print(f"Cross 점수: {format_score(result.score_cross)}")
    print(f"X 점수: {format_score(result.score_x)}")
    print(f"Cross 거리: {format_distance(result.distance_cross)}")
    print(f"X 거리: {format_distance(result.distance_x)}")
    expected = result.expected if result.expected is not None else "N/A"
    print(f"판정: {result.predicted} | expected: {expected} | {result.status}")
    if result.status == "FAIL":
        print(f"사유: {result.reason}")


def build_mode1_performance(pattern: Matrix, filter_a: Matrix) -> List[PerformanceRow]:
    """
    모드 1은 3x3 한 종류만 측정하면 되므로,
    성능 표도 1행만 만들어서 돌려준다.
    """
    average_ms = benchmark_mac(pattern, filter_a)
    return [PerformanceRow(size=3, average_ms=average_ms, operation_count=9)]


def build_mode2_performance(
    filters_by_size: Dict[int, Dict[str, Matrix]],
    benchmark_patterns: Dict[int, Matrix],
) -> List[PerformanceRow]:
    """
    모드 2 성능 분석용 표 데이터를 만든다.

    3x3은 JSON에 없을 수 있으므로 기본 Cross 패턴을 사용한다.
    5/13/25는 가능하면 실제 JSON 패턴 하나를 재사용하고,
    적절한 패턴이 없으면 Cross 필터 자체를 대체 입력으로 사용한다.
    """
    rows: List[PerformanceRow] = []

    benchmark_3x3 = default_cross_3x3()
    rows.append(
        PerformanceRow(
            size=3,
            average_ms=benchmark_mac(benchmark_3x3, benchmark_3x3),
            operation_count=9,
        )
    )

    for size in JSON_SIZES:
        cross_filter = filters_by_size.get(size, {}).get("Cross")
        if cross_filter is None:
            # 필터가 없으면 시간을 잴 수 없으므로 average_ms는 None으로 둔다.
            rows.append(
                PerformanceRow(size=size, average_ms=None, operation_count=size * size)
            )
            continue

        # 같은 크기의 실제 패턴이 하나라도 있으면 그걸 우선 사용하고,
        # 없으면 필터 자체를 임시 패턴처럼 사용한다.
        pattern = benchmark_patterns.get(size, cross_filter)
        rows.append(
            PerformanceRow(
                size=size,
                average_ms=benchmark_mac(pattern, cross_filter),
                operation_count=size * size,
            )
        )

    return rows


def run_user_input_mode() -> None:
    """
    모드 1: 사용자가 직접 3x3 필터와 패턴을 입력하는 흐름이다.

    순서:
    1. 필터 A 입력
    2. 필터 B 입력
    3. 저장 확인 출력
    4. 패턴 입력
    5. MAC 결과 계산 및 판정
    6. 3x3 성능 분석 출력
    """
    print_section(1, "필터 입력")
    filter_a = prompt_matrix("필터 A", 3)
    print()
    filter_b = prompt_matrix("필터 B", 3)

    print_section(2, "저장 확인")
    print("필터 A")
    print_matrix(filter_a)
    print()
    print("필터 B")
    print_matrix(filter_b)

    print_section(3, "패턴 입력")
    pattern = prompt_matrix("패턴", 3)

    print_section(4, "MAC 결과")
    score_a = compute_mac(pattern, filter_a)
    score_b = compute_mac(pattern, filter_b)
    pattern_sum = compute_pattern_sum(pattern)
    distance_a = compute_distance(score_a, pattern_sum)
    distance_b = compute_distance(score_b, pattern_sum)
    performance_row = build_mode1_performance(pattern, filter_a)[0]

    print(f"A 점수: {format_score(score_a)}")
    print(f"B 점수: {format_score(score_b)}")
    print(f"A 거리: {format_distance(distance_a)}")
    print(f"B 거리: {format_distance(distance_b)}")
    print(f"연산 시간(평균/{BENCHMARK_REPEATS}회): {format_ms(performance_row.average_ms)} ms")

    # 모드 1은 거리 기준으로 A / B / 판정 불가 형태를 보여준다.
    if abs(distance_a - distance_b) < EPSILON:
        print(f"판정: 판정 불가 (|A거리-B거리| < {EPSILON})")
    elif distance_a < distance_b:
        print("판정: A")
    else:
        print("판정: B")

    print_section(5, "성능 분석")
    print_performance_table([performance_row])


def run_json_mode(project_root: Path) -> None:
    """
    모드 2: data.json을 읽어서 여러 케이스를 자동 분석하는 흐름이다.

    순서:
    1. data.json 로드
    2. 필터 로드 및 검증
    3. 각 패턴 케이스 파싱/평가
    4. 성능 분석 표 출력
    5. 전체 요약 출력
    """
    json_path = project_root / "data.json"
    data = load_json_root(json_path)
    if data is None:
        return

    print_section(1, "필터 로드")
    filters_by_size, filter_messages, filter_issues_by_size = load_filters(data["filters"])
    for message in filter_messages:
        print(message)

    print_section(2, "패턴 분석 (라벨 정규화 적용)")
    results: List[CaseResult] = []
    benchmark_patterns: Dict[int, Matrix] = {}

    for case_id, case_data in data["patterns"].items():
        # 먼저 JSON 구조가 맞는지 확인한다.
        parsed_case, failure = parse_json_case(case_id, case_data)
        if failure is not None:
            results.append(failure)
            print_case_result(failure)
            continue

        # 구조가 정상이면 실제 계산까지 진행한다.
        result = evaluate_case(parsed_case, filters_by_size, filter_issues_by_size)
        results.append(result)
        print_case_result(result)

        # 성능 분석용으로 크기별 대표 패턴 하나씩만 저장해 둔다.
        if parsed_case.size not in benchmark_patterns and result.score_cross is not None:
            benchmark_patterns[parsed_case.size] = parsed_case.pattern

    print_section(3, f"성능 분석 (평균/{BENCHMARK_REPEATS}회)")
    performance_rows = build_mode2_performance(filters_by_size, benchmark_patterns)
    print_performance_table(performance_rows)

    print_section(4, "결과 요약")
    total_count = len(results)
    pass_count = sum(1 for result in results if result.status == "PASS")
    fail_results = [result for result in results if result.status == "FAIL"]

    print(f"총 테스트: {total_count}개")
    print(f"통과: {pass_count}개")
    print(f"실패: {len(fail_results)}개")

    if fail_results:
        print()
        print("실패 케이스:")
        for result in fail_results:
            print(f"- {result.id}: {result.reason}")


def main() -> None:
    """프로그램의 시작점(entry point)이다."""
    print_banner()
    mode = prompt_mode()
    if mode == "1":
        run_user_input_mode()
    else:
        run_json_mode(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
